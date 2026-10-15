"""Slide-scale schematic diagrams for vol1 ch08 (Model Training).

Simplified redraws of chapter mechanisms (input pipeline, prefetch overlap,
profiling triage, gradient accumulation, pipeline bubble) at slide font sizes.
Schematic durations are labeled as such; no measured values are implied.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

OUT = sys.argv[1]
C = dict(blue="#006395", blueL="#D1E6F3", orange="#E67817", orangeL="#FCE4CC",
         green="#008F45", greenL="#D4EFDF", violet="#7E317B", violetL="#E6D4E5",
         red="#CB202D", redL="#F5D2D5", ink="#333333", gray="#666666",
         grid="#CCCCCC", brown="#78492A", brownL="#E3D3C8", crimson="#A51C30")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"],
    "font.size": 16, "svg.fonttype": "none", "axes.edgecolor": C["gray"],
    "xtick.labelsize": 15, "ytick.labelsize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def box(ax, x, y, w, h, text, edge, fill, fs=15, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=edge, lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=C["ink"], fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, col=None, text=None, tdy=0.12):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=col or C["gray"], lw=2, mutation_scale=18))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + tdy, text, ha="center", fontsize=13, color=C["gray"])


def canvas(w, h, xl, yl):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.axis("off")
    return fig, ax


# 1. Input pipeline: three stages, min of rates ------------------------------
fig, ax = canvas(9.5, 2.2, (0, 10), (0, 2.2))
box(ax, 0.1, 0.55, 2.9, 1.3, "Prepare\n(storage + CPU)", C["green"], C["greenL"])
box(ax, 3.55, 0.55, 2.9, 1.3, "Transfer\n(PCIe)", C["violet"], C["violetL"])
box(ax, 7.0, 0.55, 2.9, 1.3, "Compute\naccelerator", C["orange"], C["orangeL"])
arrow(ax, 3.0, 1.2, 3.55, 1.2)
arrow(ax, 6.45, 1.2, 7.0, 1.2)
ax.text(1.55, 0.2, "decode, augment, tokenize", ha="center", fontsize=13, color=C["gray"])
ax.text(5.0, 0.2, "bytes per batch / link BW", ha="center", fontsize=13, color=C["gray"])
ax.text(8.45, 0.2, "FLOPs per batch / sustained rate", ha="center", fontsize=13, color=C["gray"])
save(fig, "pipeline-stages")

# 2. Serial vs overlapped prefetch (schematic) ------------------------------
prep, comp, nb = 20, 15, 4
fig, axs = plt.subplots(2, 1, figsize=(6.4, 3.4), sharex=True)
for ax, mode in zip(axs, ["serial", "overlapped"]):
    t_prep, t_comp = [], []
    if mode == "serial":
        t = 0
        for _ in range(nb):
            t_prep.append(t); t += prep
            t_comp.append(t); t += comp
    else:
        cend = 0
        for i in range(nb):
            t_prep.append(i * prep)
            s = max((i + 1) * prep, cend)
            t_comp.append(s); cend = s + comp
    for s in t_prep:
        ax.barh(1, prep, left=s, color=C["green"], edgecolor="white", height=0.6)
    for s in t_comp:
        ax.barh(0, comp, left=s, color=C["orange"], edgecolor="white", height=0.6)
    end = t_comp[-1] + comp
    ax.axvline(end, color=C["ink"], lw=1.5, ls=":")
    ax.text(end + 2, 0.5, f"{mode}: done at {end}", fontsize=14, va="center", fontweight="bold")
    ax.set_yticks([0, 1], ["accel.", "host"])
    ax.set_ylim(-0.6, 1.6)
    ax.spines["left"].set_visible(False)
axs[1].set_xlim(0, 185)
axs[1].set_xlabel("time (schematic units: prepare 20, compute 15)", fontsize=14)
save(fig, "prefetch-overlap")

# 3. Profiling triage (horizontal) ------------------------------------------
fig, ax = canvas(9.5, 3.3, (0, 10), (-0.15, 3.4))


def diamond(ax, cx, cy, w, h, text):
    ax.add_patch(Polygon([(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2)],
                         closed=True, fc="white", ec=C["ink"], lw=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=14, linespacing=1.1)


box(ax, 0.05, 1.35, 1.3, 0.8, "Profile\nthe step", C["ink"], "#F2F2F2", fs=14, bold=True)
diamond(ax, 2.55, 1.75, 2.3, 1.4, "idle\ngaps?")
diamond(ax, 5.25, 1.75, 2.3, 1.4, "memory\nnear limit?")
box(ax, 7.0, 1.3, 2.95, 0.9, "kernel counters:\nbytes, math, launches", C["blue"], C["blueL"], fs=13)
box(ax, 1.4, 0.0, 2.3, 0.75, "trace input, host,\nsynchronization", C["green"], C["greenL"], fs=13)
box(ax, 4.05, 0.0, 2.4, 0.75, "precision, checkpoint,\naccumulate", C["orange"], C["orangeL"], fs=13)
arrow(ax, 1.35, 1.75, 1.4, 1.75)
arrow(ax, 3.7, 1.75, 4.1, 1.75, text="no")
arrow(ax, 6.4, 1.75, 7.0, 1.75, text="no")
arrow(ax, 2.55, 1.05, 2.55, 0.75, text=None)
ax.text(2.7, 0.88, "yes", fontsize=13, color=C["gray"])
arrow(ax, 5.25, 1.05, 5.25, 0.75)
ax.text(5.4, 0.88, "yes", fontsize=13, color=C["gray"])
ax.text(5.0, 3.15, "after any fix: re-profile, because the bottleneck moves",
        ha="center", fontsize=14, color=C["crimson"], fontweight="bold")
save(fig, "triage-flow")

# 4. Gradient accumulation ---------------------------------------------------
fig, ax = canvas(6.4, 3.5, (0, 7), (0, 3.8))
for j, y in enumerate([2.75, 1.6, 0.45]):
    box(ax, 0.05, y, 1.8, 0.75, f"micro-batch {j+1}", C["green"], C["greenL"], fs=14)
    box(ax, 2.35, y, 1.7, 0.75, "fwd + bwd", C["orange"], C["orangeL"], fs=14)
    arrow(ax, 1.85, y + 0.375, 2.35, y + 0.375)
    arrow(ax, 4.05, y + 0.375, 4.75, 1.95, col=C["violet"])
    ax.text(4.1, [y + 0.75, y + 0.45, y - 0.12][j], f"g{j+1}", fontsize=15, color=C["violet"], fontweight="bold")
ax.add_patch(plt.Circle((5.05, 1.95), 0.32, fc=C["violetL"], ec=C["violet"], lw=2))
ax.text(5.05, 1.95, "Σ", ha="center", va="center", fontsize=20, color=C["violet"])
arrow(ax, 5.37, 1.95, 5.75, 1.95)
box(ax, 5.75, 1.45, 1.2, 1.0, "one\nstep", C["blue"], C["blueL"], fs=15, bold=True)
save(fig, "gradient-accumulation")

# 5. Pipeline bubble: naive vs micro-batched (forward pass) ------------------
fig, axs = plt.subplots(1, 2, figsize=(9.5, 2.9), sharey=True)
N, M = 4, 4
cols = [C["orange"], C["blue"], C["green"], C["violet"]]
for ax, mb, title in zip(axs, [1, M], ["one batch: utilization 1/4",
                                       "4 micro-batches: bubble 3 of 7 slots"]):
    for d in range(N):
        for m in range(mb):
            ax.barh(d, 1, left=d + m, color=cols[m], edgecolor="white", height=0.7)
            if mb > 1:
                ax.text(d + m + 0.5, d, str(m + 1), ha="center", va="center",
                        color="white", fontsize=13, fontweight="bold")
    ax.set_xlim(0, N + M - 1)
    ax.set_xticks(range(0, N + M))
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("time slot")
    ax.invert_yaxis()
axs[0].set_yticks(range(N), [f"device {d+1}" for d in range(N)])
save(fig, "pipeline-bubble")
print("ok")
