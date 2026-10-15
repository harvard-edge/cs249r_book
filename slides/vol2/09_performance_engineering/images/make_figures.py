"""Slide figures for vol2 ch09 (Fleet Efficiency).

Values are the rendered chapter's computed outputs (feat/vol2-fleet-efficiency
@ a0b79b884b). Sized for 16:9 Beamer slots: column figures 5.4 x 3.6 in, wide
strips 9.5 x 2.4-2.8 in, fonts 15-16 pt. Writes SVG with text kept as text.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = sys.argv[1]
C = dict(blue="#006395", blueL="#D1E6F3", orange="#E67817", orangeL="#FCE4CC",
         green="#008F45", greenL="#D4EFDF", violet="#7E317B", violetL="#E6D4E5",
         red="#CB202D", redL="#F5D2D5", ink="#333333", gray="#666666", grayL="#E6E6E6",
         grid="#CCCCCC", brown="#78492A", brownL="#E3D3C8", crimson="#A51C30")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"],
    "font.size": 16, "axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 14, "axes.edgecolor": C["gray"], "axes.labelcolor": C["ink"],
    "xtick.color": C["ink"], "ytick.color": C["ink"], "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})
# Factor colors: scheduling = structural brown, runtime = data-path green,
# program = compute orange, useful = blue. Kept identical on every slide.
FS, FR, FP, FU = C["brown"], C["green"], C["orange"], C["blue"]


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def stacked(ax, y, parts, h=0.62, fs=15):
    x = 0
    for w, lab, col in parts:
        ax.barh(y, w, left=x, color=col, edgecolor="white", height=h)
        if lab:
            ax.text(x + w / 2, y, lab, ha="center", va="center", color="white",
                    fontsize=fs, fontweight="bold")
        x += w


# 1. Capacity funnel (conceptual, three boundaries) ---------------------------
fig, ax = plt.subplots(figsize=(9.5, 3.0))
ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(-0.45, 3.0)
stages = [("fleet\ncapacity", 1.00, C["grayL"]), ("held,\ncomplete", 0.80, C["brownL"]),
          ("retained\nsteps", 0.62, C["greenL"]), ("at the ideal\nrate", 0.44, C["blueL"])]
edges = [C["gray"], FS, FR, FU]
for i, ((lab, hgt, fill), ec) in enumerate(zip(stages, edges)):
    x0 = 0.1 + i * 2.55
    h = 2.1 * hgt
    ax.add_patch(FancyBboxPatch((x0, 1.6 - h / 2), 1.9, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fill, ec=ec, lw=2))
    ax.text(x0 + 0.95, 1.6, lab, ha="center", va="center", fontsize=15, linespacing=1.1)
gates = [("scheduling\ngoodput S", FS), ("runtime\ngoodput R", FR), ("program\ngoodput P", FP)]
for i, (lab, col) in enumerate(gates):
    xa = 0.1 + i * 2.55 + 1.9
    ax.annotate("", xy=(xa + 0.65, 1.6), xytext=(xa, 1.6),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5, mutation_scale=20))
    ax.text(xa + 0.32, -0.05, lab, ha="center", fontsize=13, color=col, fontweight="bold",
            linespacing=1.05)
save(fig, "capacity-funnel")

# 2. Llama 3 405B split of held capacity ------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 1.9))
stacked(ax, 0, [(10, "", FR), (53, "program 53%", FP), (37, "useful 37%", FU)], h=0.7, fs=16)
ax.annotate("runtime 10%", xy=(5, 0.36), xytext=(12, 0.9), fontsize=14, color=FR, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=FR))
ax.set_xlim(0, 100); ax.set_ylim(-0.5, 1.2); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("% of the capacity Llama 3 405B held (dedicated: S = 1, R ≈ 0.90, P ≈ 0.41)")
save(fig, "llama3-split")

# 3. Shared pool snapshot -----------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 2.2))
stacked(ax, 0, [(60, "complete allocations", C["blue"]),
                (5, "", C["green"]),
                (15, "partial\ngangs", C["brown"]), (20, "unassigned", C["gray"])], h=0.7, fs=13)
ax.annotate("input wait 5%\n(runtime loss)", xy=(62.5, 0.36), xytext=(52, 0.95), fontsize=13,
            color=C["green"], ha="center", arrowprops=dict(arrowstyle="->", color=C["green"]))
ax.text(32.5, -0.62, "S = 0.65 (complete allocations)", ha="center", fontsize=16, fontweight="bold", color=C["blue"])
ax.set_xlim(0, 100); ax.set_ylim(-0.85, 1.35); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("% of the pool's in-service capacity (constructed snapshot)")
save(fig, "pool-snapshot")

# 4. Runtime loss terms --------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.6))
terms = [("input waits", 7.7), ("preemption", 1.4), ("checkpoint", 0.4),
         ("startup", 0.05), ("recovery", 0.03)]
y = np.arange(len(terms))
ax.barh(y, [t[1] for t in terms], color=[C["green"], C["brown"], C["violet"], C["gray"], C["red"]],
        height=0.62)
for yi, (_, v) in zip(y, terms):
    ax.text(v + 0.12, yi, f"{v:g}%", va="center", fontsize=15, fontweight="bold")
ax.set_yticks(y, [t[0] for t in terms])
ax.invert_yaxis()
ax.set_xlim(0, 9.5)
ax.set_xlabel("% of held time (sum ≈ 10%, R ≈ 0.90)")
save(fig, "runtime-terms")

# 5. Node vs fleet step --------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 2.5))
ax.barh(1, 180, color=C["orange"], height=0.6)
ax.barh(0, 180, color=C["orange"], height=0.6)
ax.barh(0, 65, left=180, color=C["violet"], height=0.6)
ax.text(90, 1, "180 ms", ha="center", va="center", color="white", fontsize=16, fontweight="bold")
ax.text(90, 0, "node's work 180 ms", ha="center", va="center", color="white", fontsize=16, fontweight="bold")
ax.text(212, 0, "+65", ha="center", va="center", color="white", fontsize=15, fontweight="bold")
ax.text(190, 1, "MFU 65%", va="center", fontsize=16, fontweight="bold")
ax.text(255, 0, "MFU 48%", va="center", fontsize=16, fontweight="bold")
ax.set_yticks([0, 1], ["128 GPUs", "one node (8)"])
ax.set_xlim(0, 320)
ax.set_xlabel("ms per step, same work per device")
save(fig, "node-vs-fleet")

# 6. Running example: where pool capacity went -------------------------------
fig, ax = plt.subplots(figsize=(9.5, 2.0))
stacked(ax, 0, [(35, "scheduling 35%", FS), (6, "", FR), (31, "program 31%", FP),
                (28, "useful 28%", FU)], h=0.7, fs=16)
ax.annotate("runtime 6%", xy=(38, 0.36), xytext=(38, 0.85), fontsize=14, color=FR, ha="center",
            fontweight="bold", arrowprops=dict(arrowstyle="->", color=FR))
ax.set_xlim(0, 100); ax.set_ylim(-0.5, 1.2); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("% of the pool's capacity  (S 0.65 × R 0.90 × P 0.48 = MPG 0.28)")
save(fig, "pool-loss-shares")

# 7. Lever value at each owner's ceiling ---------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.6))
labs = ["today", "scheduling\nto 0.80", "input waits\nremoved", "program\nto 0.65"]
vals = [0.28, 0.35, 0.31, 0.38]
cols = [C["gray"], FS, FR, FP]
ax.bar(range(4), vals, color=cols, width=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + 0.008, f"{v:.2f}", ha="center", fontsize=16, fontweight="bold")
ax.text(1, 0.15, "+23%", ha="center", color="white", fontsize=15, fontweight="bold")
ax.text(2, 0.15, "+9%", ha="center", color="white", fontsize=15, fontweight="bold")
ax.text(3, 0.15, "+35%", ha="center", color="white", fontsize=15, fontweight="bold")
ax.set_xticks(range(4), labs, fontsize=13)
ax.set_ylabel("pool MPG")
ax.set_ylim(0, 0.44)
save(fig, "lever-ceilings")

# 8. Localization ladder --------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 3.0))
ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3.1)
rungs = [("1  Scheduler records", "every job, always on, free", "settles S", FS, C["brownL"]),
         ("2  Job timeline", "every job, a few fields per step", "settles R", FR, C["greenL"]),
         ("3  Step traces", "chosen ranks, short windows, costly", "settles P and its term", FP, C["orangeL"])]
for i, (a, b, c, ec, fc) in enumerate(rungs):
    y0 = 2.15 - i * 1.0
    x0 = 0.1 + i * 0.6
    ax.add_patch(FancyBboxPatch((x0, y0), 6.3, 0.78, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=2))
    ax.text(x0 + 0.15, y0 + 0.5, a, fontsize=16, fontweight="bold", va="center")
    ax.text(x0 + 0.15, y0 + 0.2, b, fontsize=13, va="center", color=C["ink"])
    ax.text(x0 + 6.5, y0 + 0.39, c, fontsize=15, va="center", color=ec, fontweight="bold")
save(fig, "localization-ladder")

# 9. Drill-down hierarchy --------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.axis("off"); ax.set_xlim(0, 6); ax.set_ylim(0, 4.2)
lv = [("fleet measures", "step time, MFU, per-rank timers"),
      ("distributed traces", "lagging rank? exposed collective?"),
      ("kernel timelines", "launch gaps, stream overlap"),
      ("hardware counters", "bandwidth, occupancy, mix")]
for i, (a, b) in enumerate(lv):
    w = 5.8 - i * 0.6
    x0 = (6 - w) / 2
    y0 = 3.15 - i * 1.02
    ax.add_patch(FancyBboxPatch((x0, y0), w, 0.85, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=C["orangeL"] if i else C["grayL"], ec=C["orange"] if i else C["gray"], lw=2))
    ax.text(3, y0 + 0.56, a, ha="center", fontsize=15, fontweight="bold")
    ax.text(3, y0 + 0.22, b, ha="center", fontsize=12.5)
save(fig, "drill-hierarchy")

# 10. Compilation amortization ------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.4))
labs = ["shared artifact,\n128 replicas", "one replica"]
secs = [12, 26 * 60]
ax.barh([0, 1], secs, color=[C["blue"], C["orange"]], height=0.55)
ax.set_xscale("log")
ax.text(14, 0, "12 s", va="center", fontsize=16, fontweight="bold")
ax.text(26 * 60 * 1.1, 1, "26 min", va="center", fontsize=16, fontweight="bold")
ax.set_yticks([0, 1], labs)
ax.set_xlim(5, 10000)
ax.set_xlabel("generation time to repay a 15-min build (log)")
save(fig, "compile-amortization")
print("ok")
