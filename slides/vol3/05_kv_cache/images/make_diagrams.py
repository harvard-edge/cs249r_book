"""Slide-scale schematic diagrams for vol3 ch05 (KV Cache Management).

Simplified redraws of chapter mechanisms (context zones, the two kinds of
unused capacity, block-table indirection, forked block tables, tail
copy-on-write, the three reuse gates, radix prefix lookup, and leaf-first
eviction) at slide font sizes. Token counts and block IDs are the chapter's
own worked examples; no measured values are implied.

Usage: python3 make_diagrams.py <outdir>
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = sys.argv[1]
C = dict(blue="#006395", blueL="#D1E6F3", orange="#E67817", orangeL="#FCE4CC",
         green="#008F45", greenL="#D4EFDF", violet="#7E317B", violetL="#E6D4E5",
         red="#CB202D", redL="#F5D2D5", ink="#333333", gray="#666666",
         grid="#CCCCCC", brown="#78492A", brownL="#E3D3C8", crimson="#A51C30",
         pale="#EEEEEE")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"],
    "font.size": 16, "svg.fonttype": "none",
})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def canvas(w, h, xl, yl):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, edge, fill, fs=15, bold=False, ls="-", tc=None, lw=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fill, ec=edge, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or C["ink"], fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, col=None, lw=2, style="-|>", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=col or C["gray"], lw=lw,
                                mutation_scale=18, linestyle=ls, shrinkA=0, shrinkB=0))


def bracket(ax, x0, x1, y, text, col, fs=15, sub=None):
    ax.plot([x0, x0, x1, x1], [y + 0.12, y, y, y + 0.12], color=col, lw=1.8)
    ax.text((x0 + x1) / 2, y - 0.12, text, ha="center", va="top", fontsize=fs,
            color=col, fontweight="bold")
    if sub:
        ax.text((x0 + x1) / 2, y - 0.55, sub, ha="center", va="top", fontsize=14, color=C["gray"])


# 1. Context zones of the r18 request ------------------------------------------
zones = [("sys", 2048, C["greenL"]), ("tools", 4096, C["greenL"]),
         ("workspace", 8192, C["greenL"]), ("turn history", 16384, "white"),
         ("obs", 2048, C["blueL"])]
fig, ax = canvas(8.0, 1.75, (0, 10), (-0.75, 1.55))
scale = 10 / 32768
x = 0
edges = []
for name, n, fill in zones:
    w = n * scale
    ax.add_patch(Rectangle((x, 0.3), w, 0.8, fc=fill,
                           ec=C["green"] if fill != C["blueL"] else C["blue"], lw=2))
    ax.text(x + w / 2, 0.7, name, ha="center", va="center", fontsize=14,
            color=C["ink"], fontweight="normal" if name == "obs" else "bold")
    edges.append((x, x + w))
    x += w
ax.text(0, 1.25, "2,048 + 4,096 + 8,192 tokens", fontsize=14, color=C["gray"])
ax.text(edges[3][0] + 2.5, 1.25, "16,384", fontsize=14, color=C["gray"], ha="center")
ax.text(10, 1.25, "2,048", fontsize=14, color=C["gray"], ha="right")
for x0, x1, lab, col, ha in [(0.02, edges[2][1] - 0.04, "shared prefix: 43.8%", C["green"], "center"),
                              (edges[3][0] + 0.04, edges[3][1] - 0.04, "trajectory state", C["ink"], "center"),
                              (edges[4][0] + 0.02, 9.98, "active turn", C["blue"], "right")]:
    ax.plot([x0, x0, x1, x1], [0.27, 0.15, 0.15, 0.27], color=col, lw=1.8)
    ax.text((x0 + x1) / 2 if ha == "center" else 10, 0.05, lab, ha=ha, va="top", fontsize=15,
            color=col, fontweight="bold")
save(fig, "context-zones")

# 2. Two kinds of unused capacity ------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(8.0, 1.6), gridspec_kw=dict(width_ratios=[1, 1.1]))
ax = axs[0]
ax.set_xlim(0, 10); ax.set_ylim(-0.75, 1.6); ax.axis("off")
ax.text(0, 1.3, "internal: inside a reservation", fontsize=15, fontweight="bold")
ax.add_patch(Rectangle((0, 0.2), 10, 0.8, fc=C["pale"], ec=C["gray"], lw=2, hatch="//"))
ax.add_patch(Rectangle((0, 0.2), 10 * 2048 / 32768, 0.8, fc=C["green"], ec=C["green"], lw=2))
ax.text(0, -0.1, "2,048 of 32,768 slots used (6.25%)", fontsize=14, color=C["ink"], va="top")
ax = axs[1]
ax.set_xlim(0, 11); ax.set_ylim(-0.75, 1.6); ax.axis("off")
ax.text(0, 1.3, "external: between allocations", fontsize=15, fontweight="bold")
xs = 0
for kind, w in [("live", 1.6), ("free", 2.0), ("live", 1.0), ("free", 2.0), ("live", 1.6), ("free", 2.0)]:
    if kind == "live":
        ax.add_patch(Rectangle((xs, 0.2), w, 0.8, fc=C["blueL"], ec=C["blue"], lw=2))
    else:
        ax.add_patch(Rectangle((xs, 0.2), w, 0.8, fc="white", ec=C["gray"], lw=2, ls="--"))
        ax.text(xs + w / 2, 0.6, "256", ha="center", va="center", fontsize=15)
    xs += w
ax.text(0, -0.1, "768 free, yet a 512-slot request fails", va="top", fontsize=14,
        color=C["red"], fontweight="bold")
fig.subplots_adjust(wspace=0.15)
save(fig, "two-wastes")

# 3. Block table indirection (positions 0-9, blocks of 4) -------------------------
fig, ax = canvas(7.6, 2.6, (0, 15), (-0.1, 5.0))
OX = 2.6
ax.text(0, 4.2, "sequence", fontsize=15, color=C["gray"], va="center")
ax.text(0, 2.62, "block table", fontsize=15, color=C["gray"], va="center")
ax.text(0, 0.57, "HBM", fontsize=15, color=C["gray"], va="center")
for lb in range(3):
    x0 = OX + lb * 2.3
    for sl in range(4):
        pos = lb * 4 + sl
        filled = pos <= 9
        ax.add_patch(Rectangle((x0 + sl * 0.5, 3.9), 0.5, 0.6,
                               fc=C["greenL"] if filled else "white",
                               ec=C["ink"] if pos == 9 else C["green"], lw=2.6 if pos == 9 else 1.5,
                               ls="-" if filled else ":"))
        if filled:
            ax.text(x0 + sl * 0.5 + 0.25, 4.2, str(pos), ha="center", va="center", fontsize=14)
    ax.text(x0 + 1.0, 4.7, f"logical {lb}", ha="center", fontsize=14, color=C["gray"])
for i, pb in enumerate([3, 7, 9]):
    box(ax, OX + 0.55 + i * 2.3, 2.3, 0.9, 0.65, str(pb), C["ink"], "white", fs=17, bold=True)
    arrow(ax, OX + 1.0 + i * 2.3, 3.85, OX + 1.0 + i * 2.3, 3.0, col=C["gray"])
owner = {3: "me", 7: "me", 9: "me", 0: "o", 1: "o", 5: "o", 6: "o", 10: "o", 11: "o"}
for pb in range(12):
    x0 = OX + pb * 1.03
    kind = owner.get(pb)
    fc = C["greenL"] if kind == "me" else (C["pale"] if kind == "o" else "white")
    ec = C["green"] if kind == "me" else C["gray"]
    ax.add_patch(Rectangle((x0, 0.25), 0.88, 0.65, fc=fc, ec=ec, lw=2 if kind == "me" else 1.2))
    ax.text(x0 + 0.44, 0.57, str(pb), ha="center", va="center", fontsize=14,
            fontweight="bold" if kind == "me" else "normal")
for i, pb in enumerate([3, 7, 9]):
    arrow(ax, OX + 1.0 + i * 2.3, 2.25, OX + 0.44 + pb * 1.03, 0.95,
          col=C["ink"] if pb == 9 else C["green"], lw=2.6 if pb == 9 else 1.6)
ax.text(15, 4.75, "position 9", ha="right", fontsize=15, fontweight="bold", va="center")
ax.text(15, 3.05, "block: 9 div 4 = 2\ntable[2] = block 9\nslot: 9 mod 4 = 1", ha="right",
        fontsize=15, color=C["ink"], va="center", linespacing=1.35)
save(fig, "block-table")

# 4. Forked candidates: copy vs share -------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(8.2, 2.3))
for ax in axs:
    ax.set_xlim(0, 10); ax.set_ylim(-0.4, 5.0); ax.axis("off")
ax = axs[0]
ax.text(0, 4.6, "contiguous: copy the prefix", fontsize=15, fontweight="bold")
for r in range(4):
    y = 3.4 - r * 0.95
    ax.add_patch(Rectangle((0, y), 5.0, 0.7, fc=C["orangeL"], ec=C["orange"], lw=1.8))
    ax.text(2.5, y + 0.35, "prefix copy", ha="center", va="center", fontsize=14)
    ax.add_patch(Rectangle((5.1, y), 1.0, 0.7, fc=C["blueL"], ec=C["blue"], lw=1.8))
ax.text(6.5, 2.0, "4 × 2.7 GB\n= 10.7 GB\nwritten", fontsize=15, color=C["orange"],
        fontweight="bold", va="center")
ax = axs[1]
ax.text(0, 4.6, "paged: share the blocks", fontsize=15, fontweight="bold")
ax.add_patch(Rectangle((2.7, 1.5), 4.6, 1.5, fc=C["greenL"], ec=C["green"], lw=2))
ax.text(5.0, 2.25, "prefix blocks\nstored once", ha="center", va="center", fontsize=14)
for r in range(4):
    y = 3.4 - r * 0.95
    ax.add_patch(Rectangle((0, y), 0.9, 0.7, fc="white", ec=C["ink"], lw=1.8))
    ax.text(0.45, y + 0.35, "ABCD"[r], ha="center", va="center", fontsize=14, fontweight="bold")
    arrow(ax, 0.95, y + 0.35, 2.65, 1.6 + r * 0.43, col=C["green"], lw=1.4)
    ax.add_patch(Rectangle((8.4, y), 1.0, 0.7, fc=C["blueL"], ec=C["blue"], lw=1.8))
ax.text(0.45, -0.25, "tables", ha="center", fontsize=14, color=C["gray"])
ax.text(8.9, -0.25, "private tails", ha="center", fontsize=14, color=C["blue"])
ax.text(5.0, 0.85, "0 bytes copied", ha="center", fontsize=15, color=C["green"], fontweight="bold")
save(fig, "fork-share")

# 5. Tail copy-on-write ----------------------------------------------------------
fig, ax = canvas(8.0, 2.2, (0, 14.5), (-0.55, 4.3))
EX = [3.3, 5.2, 7.1]


def slots(bx, by, n_init, edge, new=None):
    for sl in range(4):
        fc = C["greenL"] if sl < n_init else "white"
        if new is not None and sl == new:
            fc = C["blueL"]
        ax.add_patch(Rectangle((bx + sl * 0.38, by), 0.38, 0.55, fc=fc, ec=edge, lw=1.5))


for yrow, name, tbl in [(3.4, "A", [3, 7, 9]), (0.1, "B", [3, 7, 14])]:
    ax.text(0.0, yrow + 0.33, f"candidate {name}", fontsize=15, fontweight="bold", va="center")
    for x0, pb in zip(EX, tbl):
        box(ax, x0, yrow, 0.85, 0.66, str(pb), C["ink"], "white", fs=15, bold=True)
for x0 in EX[:2]:
    slots(x0 - 0.33, 1.78, 4, C["green"])
    arrow(ax, x0 + 0.43, 3.36, x0 + 0.43, 2.37, col=C["green"], lw=1.5)
    arrow(ax, x0 + 0.43, 0.8, x0 + 0.43, 1.74, col=C["green"], lw=1.5)
ax.text(0.0, 2.05, "shared:\ntwo owners", fontsize=14, color=C["green"], va="center", fontweight="bold")
slots(10.2, 3.45, 2, C["green"])
slots(10.2, 0.15, 3, C["blue"], new=2)
arrow(ax, 8.0, 3.73, 10.15, 3.73, col=C["ink"], lw=1.5)
arrow(ax, 8.0, 0.43, 10.15, 0.43, col=C["blue"], lw=1.5)
ax.text(10.96, 4.15, "block 9", ha="center", fontsize=14)
ax.text(10.96, -0.35, "block 14 (B only)", ha="center", fontsize=14, color=C["blue"])
arrow(ax, 10.6, 3.4, 10.6, 0.75, col=C["violet"], lw=2.2)
ax.text(11.0, 2.05, "copy 2 entries,\nthen B writes 10", fontsize=14, color=C["violet"],
        fontweight="bold", va="center")
save(fig, "cow-tail")

# 6. Three reuse gates, and the one no cache checks ------------------------------
fig, ax = canvas(8.4, 2.0, (0, 13.6), (-0.8, 3.0))
box(ax, 0.0, 0.85, 1.9, 0.9, "matched\nprefix", C["ink"], C["pale"], fs=14, bold=True)
gates = [("compatible", "same tokens\nand config", C["green"]),
         ("available", "resident, or\npaged back in", C["blue"]),
         ("permitted", "same tenant\nor domain", C["violet"])]
for i, (g, sub, col) in enumerate(gates):
    x0 = 2.5 + i * 2.95
    box(ax, x0, 0.85, 2.4, 0.9, g, col, "white", fs=14, bold=True, tc=col)
    ax.text(x0 + 1.2, 0.6, sub, ha="center", va="top", fontsize=14, color=C["gray"], linespacing=1.1)
    arrow(ax, x0 - 0.55, 1.3, x0, 1.3)
arrow(ax, 2.5 + 3 * 2.95 - 0.55, 1.3, 11.4, 1.3)
box(ax, 11.4, 0.85, 2.15, 0.9, "reusable", C["green"], C["greenL"], fs=15, bold=True)
ax.add_patch(FancyBboxPatch((6.5, 2.2), 7.05, 0.62, boxstyle="round,pad=0.02,rounding_size=0.06",
                            fc="white", ec=C["red"], lw=2, ls="--"))
ax.text(10.0, 2.51, "current?  no cache checks this", ha="center", va="center", fontsize=15,
        color=C["red"], fontweight="bold")
save(fig, "reuse-gates")

# 7/8. Radix tree: lookup path, and leaf-first eviction ---------------------------
NODES = {"root": (5.2, 3.2, "shared prefix"), "hist": (2.3, 1.8, "repair history"),
         "other": (8.3, 1.8, "other task"), "test": (1.0, 0.4, "test-log turn"),
         "rej": (4.0, 0.4, "rejected"), "final": (8.3, 0.4, "final turn")}
EDGES = [("root", "hist"), ("root", "other"), ("hist", "test"), ("hist", "rej"), ("other", "final")]
W, H = 2.75, 0.8


def tree(ax, style):
    for a, b in EDGES:
        xa, ya, _ = NODES[a]; xb, yb, _ = NODES[b]
        st = style.get(b, "dormant")
        on = style.get(a) == "active" and st == "active"
        ax.plot([xa, xa, xb, xb], [ya - H / 2, (ya + yb) / 2, (ya + yb) / 2, yb + H / 2],
                color=C["green"] if on else C["gray"], lw=3.5 if on else 1.4,
                ls=":" if st == "evicted" else "-")
    for k, (x, y, lab) in NODES.items():
        st = style.get(k, "dormant")
        fc, ec, ls, tc = {"active": (C["greenL"], C["green"], "-", C["ink"]),
                          "dormant": ("white", C["green"], "--", C["ink"]),
                          "eligible": (C["pale"], C["ink"], "--", C["ink"]),
                          "evicted": ("white", C["grid"], ":", "#AAAAAA")}[st]
        box(ax, x - W / 2, y - H / 2, W, H, lab, ec, fc, fs=14, ls=ls, tc=tc,
            bold=st == "active")


fig, ax = canvas(7.4, 3.8, (-0.4, 10.6), (-0.95, 3.75))
tree(ax, {"root": "active", "hist": "active", "test": "active"})
box(ax, 8.0, -0.85, 2.5, 0.62, "new tokens", C["blue"], C["blueL"], fs=14, bold=True)
arrow(ax, 1.0, -0.02, 1.0, -0.54, col=C["blue"])
ax.plot([1.0, 8.0], [-0.54, -0.54], color=C["blue"], lw=2)
ax.text(4.5, -0.9, "prefill starts at the divergence point", ha="center", fontsize=14, color=C["blue"])
ax.text(10.5, 3.2, "matched path:\nKV reused", ha="right", fontsize=14, color=C["green"],
        fontweight="bold", va="center")
save(fig, "radix-lookup")

fig, ax = canvas(7.4, 3.8, (-0.4, 10.6), (-0.95, 3.75))
tree(ax, {"root": "active", "hist": "active", "test": "active", "rej": "eligible",
          "other": "dormant", "final": "eligible"})
ax.text(4.0, -0.3, "eligible now", ha="center", va="top", fontsize=14, color=C["ink"], fontweight="bold")
ax.text(8.3, -0.3, "eligible now", ha="center", va="top", fontsize=14, color=C["ink"], fontweight="bold")
ax.text(10.5, 2.55, "eligible only after\nits child is evicted", ha="right", va="bottom",
        fontsize=14, color=C["green"])
ax.text(-0.3, 3.2, "on a running\nrequest's path", fontsize=14, color=C["green"], va="center",
        fontweight="bold")
save(fig, "leaf-first-eviction")
print("ok")
