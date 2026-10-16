"""Slide-scale schematic diagrams for vol4 ch01 (The Causal Boundary).

Redraws of chapter mechanisms at slide font sizes: the aisle scene of the
stopping notebook, where the learned output lands across eras, the
command-to-motion chain, the closed causal loop, the three machine classes,
the scope test, the five-level machine, the Tempe timeline (NTSB/HAR-19/03
values as printed by the chapter), and the convergence of safety, security,
and accountability on the permission path. Role colors: blue for learned
proposals (Brain), green for the deterministic permission path (Nervous
System), brown for the physical plant (Body), red only for limits and
withheld braking.

Usage: python3 make_diagrams.py <outdir>
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

OUT = sys.argv[1]
C = dict(blue="#006395", blueL="#D1E6F3", orange="#E67817", orangeL="#FCE4CC",
         green="#008F45", greenL="#D4EFDF", violet="#7E317B", violetL="#E6D4E5",
         red="#CB202D", redL="#F5D2D5", ink="#333333", gray="#666666",
         grid="#CCCCCC", brown="#78492A", brownL="#E3D3C8", lgray="#F2F2F2")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"],
    "font.size": 15, "svg.fonttype": "none",
})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def canvas(w, h, xl, yl):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, edge, fill, fs=15, bold=False, ls="-", tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=edge, lw=2, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or C["ink"], fontweight="bold" if bold else "normal", linespacing=1.12)


def arrow(ax, x0, y0, x1, y1, col=None, lw=2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=col or C["gray"], lw=lw, mutation_scale=18))


# 1. The aisle scene of the stopping notebook --------------------------------
fig, ax = canvas(9.5, 2.5, (0, 10), (-0.9, 2.3))
ax.plot([0, 10], [0, 0], color=C["gray"], lw=2)
ax.add_patch(Rectangle((0.3, 0.22), 1.9, 0.75, fc=C["brownL"], ec=C["brown"], lw=2))
ax.plot([1.6, 1.6, 2.0], [0.97, 1.55, 1.75], color=C["brown"], lw=4, solid_capstyle="round")
for wx in (0.7, 1.8):
    ax.add_patch(Circle((wx, 0.16), 0.16, fc=C["brown"], ec="none"))
ax.text(1.25, 0.6, "loaded base", ha="center", va="center", fontsize=16, color=C["brown"], fontweight="bold")
arrow(ax, 2.4, 1.2, 3.6, 1.2, col=C["ink"], lw=2.5)
ax.text(3.75, 1.2, "1.5 m/s", va="center", fontsize=16, fontweight="bold", color=C["ink"])
ax.add_patch(Rectangle((8.6, 0.0), 1.0, 2.1, fc=C["lgray"], ec=C["gray"], lw=2))
ax.text(9.1, 1.05, "rack\nend", ha="center", va="center", fontsize=16, color=C["gray"])
ax.add_patch(Circle((8.1, 1.62), 0.14, fc=C["ink"], ec="none"))
ax.plot([8.1, 8.1], [1.48, 0.62], color=C["ink"], lw=3)
ax.plot([8.1, 7.85], [0.62, 0.02], color=C["ink"], lw=3)
ax.plot([8.1, 8.3], [0.62, 0.02], color=C["ink"], lw=3)
ax.plot([7.8, 8.1, 8.4], [1.0, 1.25, 1.0], color=C["ink"], lw=3)
ax.text(7.55, 1.75, "worker steps out", ha="right", fontsize=16, color=C["ink"])
ax.annotate("", xy=(2.2, -0.35), xytext=(7.85, -0.35),
            arrowprops=dict(arrowstyle="<->", color=C["red"], lw=2.2, mutation_scale=16))
ax.text(5.0, -0.48, "clear distance 1.10 m", ha="center", va="top", fontsize=16,
        color=C["red"], fontweight="bold")
save(fig, "aisle-scene")

# 2. Where the learned output lands, era by era ------------------------------
fig, ax = canvas(9.5, 3.0, (0, 10), (-0.15, 3.85))
rows = [("before ~1985", ["perceive", "plan", "act"], [0, 0, 0], "Unimate 1961\nShakey 1966-72"),
        ("to ~2015", ["perceive", "plan", "act"], [1, 0, 0], "ALVINN 1989\nStanley 2005"),
        ("from ~2016", ["learned policy: camera to torque"], [1], "visuomotor 2016\nchunking 2023"),
        ("2023 on", ["pretrained backbone proposes"], [2], "VLA models\n2023-25")]
for i, (era, parts, learned, ex) in enumerate(rows):
    y = 3.0 - i * 1.0
    ax.text(0.0, y + 0.35, era, va="center", fontsize=16, fontweight="bold", color=C["ink"])
    x0, wtot = 2.1, 5.0
    w = (wtot - 0.3 * (len(parts) - 1)) / len(parts)
    for j, (p, l) in enumerate(zip(parts, learned)):
        edge, fill = [(C["gray"], C["lgray"]), (C["blue"], C["blueL"]), (C["blue"], "#A9CDE4")][l]
        box(ax, x0 + j * (w + 0.3), y, w, 0.75, p, edge, fill, fs=15)
        if j:
            arrow(ax, x0 + j * (w + 0.3) - 0.3, y + 0.35, x0 + j * (w + 0.3), y + 0.35)
    ax.text(7.3, y + 0.37, ex, va="center", fontsize=15, color=C["gray"], linespacing=1.25)
save(fig, "learned-output-eras")

# 3. Command-to-motion chain ---------------------------------------------------
fig, ax = canvas(9.5, 2.5, (0, 10.75), (-0.95, 1.95))
chain = [("chunk policy\ntensor", C["blue"], C["blueL"], "--"),
         ("permission\npath", C["green"], C["greenL"], "-"),
         ("position and\ncurrent loops", C["green"], C["greenL"], "-"),
         ("PWM compare\nregister", C["green"], C["greenL"], "-"),
         ("winding drives\nthe mass", C["brown"], C["brownL"], "-")]
w, g = 1.9, 0.2
xs = [0.02 + i * (w + g) + (0.35 if i >= 4 else 0) for i in range(5)]
for i, (t, e, f, ls) in enumerate(chain):
    box(ax, xs[i], 0.25, w, 1.0, t, e, f, fs=14, ls=ls)
    if i:
        arrow(ax, xs[i - 1] + w, 0.75, xs[i], 0.75)
xb = (xs[3] + w + xs[4]) / 2
ax.plot([xb, xb], [-0.75, 1.75], color=C["red"], lw=2.5, ls="--")
ax.text(xb, 1.8, "causal boundary", ha="center", va="bottom", fontsize=16, color=C["red"], fontweight="bold")
ax.text(xb - 0.2, -0.3, "upstream: rewriting a command is free", ha="right", va="center",
        fontsize=15, color=C["green"])
ax.text(xb + 0.2, -0.3, "downstream: work\nagainst inertia", ha="left", va="center",
        fontsize=15, color=C["brown"], linespacing=1.15)
save(fig, "command-chain")

# 4. Advisory output vs the closed causal loop -------------------------------
fig, ax = canvas(5.3, 3.4, (0, 7), (-0.2, 4.2))
ax.text(0.0, 4.0, "advisory", fontsize=15, fontweight="bold", color=C["gray"], va="center")
box(ax, 0.0, 2.95, 1.9, 0.75, "model", C["gray"], C["lgray"], fs=14)
box(ax, 2.5, 2.95, 1.9, 0.75, "prediction", C["gray"], C["lgray"], fs=14)
box(ax, 5.0, 2.95, 1.95, 0.75, "person or\nfilter decides", C["gray"], C["lgray"], fs=13)
arrow(ax, 1.9, 3.32, 2.5, 3.32); arrow(ax, 4.4, 3.32, 5.0, 3.32)
ax.text(0.0, 2.45, "physical AI", fontsize=15, fontweight="bold", color=C["ink"], va="center")
box(ax, 0.0, 1.35, 1.9, 0.75, "observation", C["violet"], C["violetL"], fs=13)
box(ax, 2.5, 1.35, 1.9, 0.75, "proposal", C["blue"], C["blueL"], fs=14, ls="--")
box(ax, 5.0, 1.35, 1.95, 0.75, "permission\npath", C["green"], C["greenL"], fs=14)
box(ax, 5.0, 0.0, 1.95, 0.75, "actuators", C["brown"], C["brownL"], fs=14)
box(ax, 2.5, 0.0, 1.9, 0.75, "plant and\nworld change", C["brown"], C["brownL"], fs=13)
arrow(ax, 1.9, 1.72, 2.5, 1.72); arrow(ax, 4.4, 1.72, 5.0, 1.72)
arrow(ax, 5.97, 1.35, 5.97, 0.75); arrow(ax, 5.0, 0.37, 4.4, 0.37)
ax.annotate("", xy=(0.95, 1.35), xytext=(2.5, 0.37),
            arrowprops=dict(arrowstyle="-|>", color=C["gray"], lw=2, mutation_scale=18,
                            connectionstyle="angle,angleA=180,angleB=90"))
save(fig, "causal-loop")

# 5. Three machine classes -----------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(9.5, 3.2))
titles = [("Class 1\nmass-dominated mobility", "spends\nstopping clearance"),
          ("Class 2\ncontact-dominated", "spends force and\ndeformation margin"),
          ("Class 3\nenergy and thermal", "spends heat and\nsupply margin")]
for ax, (t1, t2) in zip(axs, titles):
    ax.set_xlim(0, 4); ax.set_ylim(-0.6, 3.7); ax.axis("off")
    ax.text(2, 3.7, t1, ha="center", va="top", fontsize=16, fontweight="bold", color=C["ink"], linespacing=1.05)
    ax.text(2, -0.6, t2, ha="center", va="bottom", fontsize=16, color=C["red"], fontweight="bold", linespacing=1.05)
a = axs[0]
a.plot([0.1, 3.9], [0.8, 0.8], color=C["gray"], lw=2)
a.add_patch(Rectangle((0.2, 1.0), 1.0, 0.5, fc=C["brownL"], ec=C["brown"], lw=2))
for wx in (0.42, 0.98):
    a.add_patch(Circle((wx, 0.92), 0.11, fc=C["brown"], ec="none"))
arrow(a, 1.3, 1.65, 2.2, 1.65, col=C["ink"])
a.text(1.75, 1.75, "v", ha="center", va="bottom", fontsize=15, fontstyle="italic")
a.add_patch(Rectangle((3.3, 0.8), 0.45, 1.6, fc=C["lgray"], ec=C["gray"], lw=2))
a.annotate("", xy=(1.2, 0.55), xytext=(3.3, 0.55),
           arrowprops=dict(arrowstyle="<->", color=C["red"], lw=2, mutation_scale=14))
a = axs[1]
a.add_patch(Rectangle((0.2, 0.7), 0.45, 0.9, fc=C["brownL"], ec=C["brown"], lw=2))
a.plot([0.43, 1.2, 2.0], [1.6, 2.35, 1.6], color=C["brown"], lw=4, solid_capstyle="round")
a.add_patch(Rectangle((2.0, 1.42), 0.3, 0.36, fc=C["brown"], ec="none"))
xs = [2.3 + 0.1 * k for k in range(9)]
a.plot(xs, [1.6 + (0.13 if k % 2 else -0.13) * (0 < k < 8) for k in range(9)], color=C["ink"], lw=2)
a.add_patch(Rectangle((3.15, 0.7), 0.35, 1.8, fc=C["lgray"], ec=C["gray"], lw=2))
a.text(2.7, 1.95, "F", ha="center", fontsize=15, fontstyle="italic", color=C["red"], fontweight="bold")
a = axs[2]
a.add_patch(Rectangle((0.35, 0.8), 1.4, 1.5, fc=C["brownL"], ec=C["brown"], lw=2))
a.plot([0.5 + 0.11 * k for k in range(11)], [1.55 + (0.35 if k % 2 else -0.35) * (0 < k < 10) for k in range(11)],
       color=C["brown"], lw=2.2)
a.add_patch(Rectangle((2.5, 0.75), 0.4, 1.6, fc="white", ec=C["gray"], lw=2))
a.add_patch(Rectangle((2.5, 0.75), 0.4, 1.15, fc=C["orange"], ec="none"))
a.plot([2.3, 3.1], [2.1, 2.1], color=C["red"], lw=2.5, ls="--")
a.text(3.2, 2.1, "limit", va="center", fontsize=15, color=C["red"])
a.text(3.0, 1.2, "T", va="center", fontsize=15, fontstyle="italic", color=C["orange"])
save(fig, "three-classes")

# 6. Scope test -----------------------------------------------------------------
fig, ax = canvas(4.7, 4.1, (-2.75, 2.75), (-2.55, 2.65))
r = 1.32
cents = [(-0.68, 0.42), (0.68, 0.42), (0.0, -0.72)]
fills = [C["blue"], C["brown"], C["green"]]
for (cx, cy), col in zip(cents, fills):
    ax.add_patch(Circle((cx, cy), r, fc=col, alpha=0.13, ec=col, lw=2.5))
ax.text(-1.75, 1.75, "learned\nmodel", ha="center", va="bottom", linespacing=1.0, fontsize=15, fontweight="bold", color=C["blue"])
ax.text(1.75, 1.75, "physical\nfeedback", ha="center", va="bottom", linespacing=1.0, fontsize=15, fontweight="bold", color=C["brown"])
ax.text(0.0, -2.42, "delegated authority", ha="center", fontsize=15, fontweight="bold", color=C["green"])
ax.text(0.0, 1.08, "advisory", ha="center", va="center", fontsize=14, color=C["ink"])
ax.text(-1.12, -0.52, "digital\nautonomy", ha="center", va="center", fontsize=14, color=C["ink"])
ax.text(1.12, -0.52, "classical\ncontrol", ha="center", va="center", fontsize=14, color=C["ink"])
ax.text(0.0, 0.04, "physical\nAI", ha="center", va="center", fontsize=16, fontweight="bold", color=C["ink"])
save(fig, "scope-venn")

# 7. The five-level machine ------------------------------------------------------
fig, ax = canvas(5.0, 4.1, (0, 10), (0.0, 6.5))
levels = [("Brain", "learned proposers", "2 Hz / 20 Hz / 40 Hz", C["blue"], C["blueL"], 4.95),
          ("Nervous System", "permission path, lease, fallback", "1 kHz", C["green"], C["greenL"], 3.45),
          ("Body", "drives, transmission, windings", "10-25 kHz", C["brown"], C["brownL"], 2.4),
          ("Boundary", "register write, sensor capture", "event-timed", C["ink"], C["lgray"], 1.35),
          ("World", "contact, payload, people", "continuous", C["violet"], C["violetL"], 0.3)]
for name, desc, rate, e, f, y in levels:
    ax.add_patch(FancyBboxPatch((0.4, y), 9.2, 0.95, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=f, ec=e, lw=2, ls="--" if name == "Brain" else "-"))
    ax.text(0.65, y + 0.68, name, va="center", fontsize=14, fontweight="bold", color=e)
    ax.text(0.65, y + 0.25, desc, va="center", fontsize=12, color=C["ink"])
    ax.text(9.4, y + 0.47, rate, ha="right", va="center", fontsize=14, color=C["ink"])
ax.plot([0.4, 9.6], [4.67, 4.67], color=C["ink"], lw=2, ls=(0, (5, 3)))
ax.text(5.0, 4.67, "proposal boundary: above may only propose", ha="center",
        va="center", fontsize=12, fontweight="bold", color=C["ink"],
        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
ax.add_patch(FancyBboxPatch((0.1, 0.05), 9.8, 6.35, boxstyle="round,pad=0.02,rounding_size=0.12",
                            fc="none", ec=C["gray"], lw=1.8, ls=(0, (4, 3))))
ax.text(0.4, 6.15, "governance: lifecycle envelope, no control rate", va="center", fontsize=12,
        color=C["gray"], fontstyle="italic")
save(fig, "five-levels")

# 8. Tempe timeline (NTSB/HAR-19/03; broken time axis) ---------------------------


def tx(t):
    return (6.0 - t) / 4.5 * 2.3 if t >= 1.5 else 2.3 + (1.5 - t) / 1.5 * 6.2


fig, ax = canvas(9.5, 2.5, (-0.1, 10.3), (-1.3, 1.85))
ax.plot([0, 2.2], [0, 0], color=C["gray"], lw=2)
ax.plot([2.4, 8.5], [0, 0], color=C["gray"], lw=2)
ax.text(8.62, 0, "impact", ha="left", va="center", fontsize=15, color=C["red"], fontweight="bold")
ax.text(2.3, 0, "//", ha="center", va="center", fontsize=15, color=C["gray"])
ax.add_patch(Rectangle((tx(1.2), 0.08), tx(0.2) - tx(1.2), 0.95, fc=C["redL"], ec=C["red"], lw=2))
ax.text((tx(1.2) + tx(0.2)) / 2, 0.555, "1 s action suppression\n18.7 m with the brake withheld",
        ha="center", va="center", fontsize=14, fontweight="bold", color=C["red"], linespacing=1.1)
marks = [(5.6, "first detected", 1, C["ink"]), (1.2, "emergency\nrecognized, 43.2 mph", -1, C["ink"]),
         (0.2, "gradual slowdown\nplanned, 40.5 mph", -1, C["ink"]),
         (0.02, "operator takes\nwheel at 0.02 s", 1, C["ink"]), (0.0, "", 0, C["red"])]
for t, lab, side, col in marks:
    x = tx(t)
    ax.plot([x, x], [-0.12, 0.12], color=col, lw=2.5)
    if t > 5:
        ax.text(x - 0.15, -0.22, f"{t:g} s before impact", ha="left", va="top", fontsize=15, color=C["gray"])
    elif t > 0.1:
        ax.text(x, -0.22, f"{t:g} s", ha="center", va="top", fontsize=15, color=C["gray"])
    if side > 0 and t < 0.1:
        ax.text(x - 0.25, 1.15, lab, ha="left", va="bottom", fontsize=15, color=col, linespacing=1.0)
        ax.plot([x, x], [0.12, 1.1], color=C["gray"], lw=1)
    elif side > 0:
        ax.text(x, 0.92, lab, ha="center", va="bottom", fontsize=15, color=col, linespacing=1.0)
    elif side < 0:
        ax.text(x, -0.62, lab, ha="center", va="top", fontsize=15, color=col, linespacing=1.0)

save(fig, "tempe-timeline")

# 9. Safety, security, and accountability converge on the permission path -------
fig, ax = canvas(9.5, 2.75, (0, 10), (0, 3.2))
left = [("safety", "slick floor, dirty camera"), ("security", "spoofed sensor data"),
        ("accountability", "reconstruct an incident")]
right = ["fresh, authenticated\nphysical state", "deterministic authority\nand a feasible fallback",
         "synchronized causal\ntelemetry"]
for i, (h, d) in enumerate(left):
    y = 2.25 - i * 1.05
    box(ax, 0.0, y, 3.1, 0.8, "", C["gray"], C["lgray"])
    ax.text(1.55, y + 0.56, h, ha="center", va="center", fontsize=16, fontweight="bold", color=C["ink"])
    ax.text(1.55, y + 0.21, d, ha="center", va="center", fontsize=15, color=C["ink"])
    arrow(ax, 3.1, y + 0.4, 3.75, 1.6)
box(ax, 3.75, 1.0, 2.2, 1.2, "permission\npath", C["green"], C["greenL"], fs=16, bold=True)
for i, t in enumerate(right):
    y = 2.25 - i * 1.05
    box(ax, 6.65, y, 3.3, 0.8, t, C["green"], "white", fs=15)
    arrow(ax, 5.95, 1.6, 6.65, y + 0.4, col=C["green"])
save(fig, "convergence")
print("ok")
