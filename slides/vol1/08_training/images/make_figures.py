"""Slide figures for vol1 ch08 (Model Training).

Values are the rendered chapter's LEGO outputs at the slides-refresh base commit
(the precision ladder, roofline, and scaling curves reuse the chapter figure
code's own inputs). Each figure is sized for its slot on a 16:9 Beamer slide so
that text lands at 8 pt or more: column figures 5.4 x 3.6 in, wide strips
9.5 x 2.6 in, all with 15-16 pt fonts. Writes SVG with text kept as text.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = sys.argv[1]
C = dict(blue="#006395", blueL="#D1E6F3", orange="#E67817", orangeL="#FCE4CC",
         green="#008F45", greenL="#D4EFDF", violet="#7E317B", violetL="#E6D4E5",
         red="#CB202D", redL="#F5D2D5", ink="#333333", gray="#666666",
         grid="#CCCCCC", brown="#78492A", brownL="#E3D3C8", crimson="#A51C30")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"],
    "font.size": 16, "axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 14, "axes.edgecolor": C["gray"], "axes.labelcolor": C["ink"],
    "xtick.color": C["ink"], "ytick.color": C["ink"], "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})
COL = (5.4, 3.6)     # column figure
WIDE = (9.5, 2.6)    # full-width strip


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# 1. 16 bytes per parameter (wide strip) -----------------------------------
fig, ax = plt.subplots(figsize=WIDE)
rows = [("Inference", [(2, "W", C["blue"])]),
        ("Training", [(2, "W", C["blue"]), (2, "G", C["violet"]),
                      (4, "FP32 master", C["orange"]), (8, "Adam m, v", C["brown"])])]
for i, (lab, segs) in enumerate(rows):
    x = 0
    for w, t, col in segs:
        ax.barh(i, w, left=x, color=col, edgecolor="white", height=0.66)
        ax.text(x + w / 2, i, t, ha="center", va="center", color="white",
                fontsize=15, fontweight="bold")
        x += w
    ax.text(x + 0.25, i, f"{x} B per parameter", va="center", fontsize=16,
            fontweight="bold", color=C["ink"])
ax.set_yticks([0, 1], [r[0] for r in rows])
ax.set_xlim(0, 22)
ax.set_xticks([0, 2, 4, 8, 16])
ax.set_xlabel("Bytes per parameter, before any activation   (W = FP16 weights, G = FP16 gradients)")
ax.set_ylim(1.5, -0.5)
save(fig, "static-state-16-bytes")

# 2. GPT-2 XL memory waterfall ---------------------------------------------
fig, ax = plt.subplots(figsize=(8, 3.6))
steps = ["FP32", "Mixed\nprecision", "+ Flash-\nAttention", "+ 2 micro-\nbatches of 16"]
total = [598, 311, 109.6, 66.8]
x = np.arange(4)
ax.bar(x, [24] * 4, color=C["orange"], width=0.6, label="static state (24 GB)")
ax.bar(x, [t - 24 for t in total], bottom=24, color=C["blue"], width=0.6, label="activations")
for xi, t in zip(x, total):
    ax.text(xi, max(t, 80) + 14, f"{t:g} GB", ha="center", fontsize=17, fontweight="bold",
            color=C["green"] if t < 80 else C["ink"])
ax.axhline(80, color=C["red"], lw=2.2, ls="--", label="H100: 80 GB")
ax.set_xticks(x, steps)
ax.set_ylabel("GB per batch-32 step")
ax.set_ylim(0, 680)
ax.legend(loc="upper right", frameon=False)
save(fig, "gpt2-memory-waterfall")

# 3. Activation memory per layer (wide strip) -------------------------------
fig, ax = plt.subplots(figsize=(9.5, 2.3))
parts = [(419.4, C["blue"]), (419.4, C["violet"]), (4194.3, C["red"]), (10, C["gray"])]
xx = 0
for w, col in parts:
    ax.barh(0, w, left=xx, color=col, edgecolor="white", height=0.6)
    xx += w
ax.text(838.8 + 4194.3 / 2, 0, "attention matrices: 4,194 MB (83%)", ha="center",
        va="center", color="white", fontsize=17, fontweight="bold")
ax.text(0, -0.5, "Q, K, V, out: 419", color=C["blue"], fontsize=15, fontweight="bold", va="top")
ax.text(1500, -0.5, "FFN: 419", color=C["violet"], fontsize=15, fontweight="bold", va="top")
ax.text(5043, 0.42, "total 5,043 MB per layer", ha="right", fontsize=15)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlim(0, 5100)
ax.set_ylim(-0.95, 0.65)
ax.set_xlabel("MB of saved activations per GPT-2 XL layer (batch 32, 1,024 tokens)")
save(fig, "activation-per-layer")

# 4. Fine-tuning scope -----------------------------------------------------
fig, ax = plt.subplots(figsize=COL)
comp = [("base weights", [16, 16], C["blue"]),
        ("grads, master, moments", [112, 0.11], C["orange"]),
        ("activations", [36.5, 36.5], C["violet"])]
bot = np.zeros(2)
for name, vals, col in comp:
    ax.bar([0, 1], vals, bottom=bot, color=col, width=0.55, label=name)
    bot += np.array(vals)
for i, t in enumerate(["≈165 GB", "≈53 GB"]):
    ax.text(i, bot[i] + 5, t, ha="center", fontsize=17, fontweight="bold",
            color=C["green"] if i else C["ink"])
ax.axhline(80, color=C["red"], lw=2.2, ls="--")
ax.text(2.85, 84, "80 GB", color=C["red"], ha="right", fontsize=15, fontweight="bold")
ax.set_xticks([0, 1], ["Full\nfine-tune", "LoRA\nadapters"])
ax.set_ylabel("GB")
ax.set_ylim(0, 190)
ax.set_xlim(-0.5, 2.9)
ax.legend(frameon=False, loc="upper right", fontsize=13)
save(fig, "llama8b-finetune-scope")

# 5. ResNet-50 feeding ------------------------------------------------------
fig, ax = plt.subplots(figsize=COL)
x = np.arange(2)
demand, supply = [3200, 10100], [3375, 5250]
ax.bar(x - 0.19, demand, 0.36, color=C["orange"], label="GPU demand")
ax.bar(x + 0.19, supply, 0.36, color=C["green"], label="host supply")
for xi, d, s in zip(x, demand, supply):
    ax.text(xi - 0.19, d + 250, f"{d/1000:.1f}k", ha="center", fontsize=15)
    ax.text(xi + 0.19, s + 250, f"{s/1000:.2f}k", ha="center", fontsize=15)
ax.text(1.19, 6500, "52%\nbusy", ha="center", va="bottom", fontsize=18, color=C["red"], fontweight="bold")
ax.text(0, 4600, "fed", ha="center", fontsize=18, color=C["green"], fontweight="bold")
ax.set_xticks(x, ["A100", "H100"])
ax.set_ylabel("images per second")
ax.set_ylim(0, 13000)
ax.set_yticks([0, 4000, 8000, 12000], ["0", "4k", "8k", "12k"])
ax.legend(frameon=False, loc="upper left")
save(fig, "resnet50-feeding")

# 6. Precision -> days -----------------------------------------------------
fig, ax = plt.subplots(figsize=COL)
days, cost = [518, 70, 35], ["$49,751", "$6,748", "$3,370"]
ax.bar(range(3), days, color=[C["gray"], C["blue"], C["orange"]], width=0.58)
for i, (d, c) in enumerate(zip(days, cost)):
    ax.text(i, d * 1.15, f"{d} d\n{c}", ha="center", fontsize=16, fontweight="bold")
ax.set_yscale("log")
ax.set_ylim(10, 3000)
ax.set_xticks(range(3), ["FP32\n67", "TF32\n494", "BF16\n989"])
ax.set_xlabel("format, with H100 peak TFLOP/s")
ax.set_ylabel("days on one H100")
save(fig, "precision-days")

# 7. sqrt(L) checkpoint curve ----------------------------------------------
fig, ax = plt.subplots(figsize=COL)
L = 48
k = np.arange(1, 49)
ax.plot(k, k + L / k, color=C["blue"], lw=3)
ax.axhline(L, color=C["gray"], ls=":", lw=2)
ax.text(47, 50, "store all 48", ha="right", fontsize=15, color=C["gray"])
ax.plot([7], [7 + L / 7], "o", color=C["crimson"], ms=12)
ax.annotate("k = 7: ≈14\n(−71%)", xy=(7, 13.9), xytext=(12, 30), fontsize=16,
            color=C["crimson"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C["crimson"], lw=1.8))
ax.set_xlabel("checkpoints k  (L = 48 layers)")
ax.set_ylabel("layers' worth stored")
ax.set_ylim(0, 56)
save(fig, "checkpoint-sqrt-l")

# 8. AllReduce share (wide strip) ------------------------------------------
fig, ax = plt.subplots(figsize=WIDE)
cases = ["full batch\nper replica", "batch split\n8 ways"]
comp_ms, sync = [596, 75], [12, 12]
for i in range(2):
    ax.barh(i, comp_ms[i], color=C["orange"], height=0.6, label="compute" if i == 0 else None)
    ax.barh(i, sync[i], left=comp_ms[i], color=C["violet"], height=0.6,
            label="AllReduce, 12 ms" if i == 0 else None)
    share = sync[i] / (comp_ms[i] + sync[i]) * 100
    ax.text(comp_ms[i] + sync[i] + 14, i, f"sync {share:.0f}% of step", va="center",
            fontsize=17, fontweight="bold", color=C["violet"])
ax.set_yticks([0, 1], cases)
ax.set_ylim(1.6, -0.6)
ax.set_xlim(0, 820)
ax.set_xlabel("ms per step on each of 8 H100s")
ax.legend(frameon=False, loc="lower right", ncol=2)
save(fig, "allreduce-share")

# 9. ZeRO stages ------------------------------------------------------------
fig, ax = plt.subplots(figsize=COL)
gb = [24, 8.2, 5.6, 3]
ax.bar(range(4), gb, color=[C["gray"], C["blue"], C["blue"], C["blue"]], width=0.6)
for i, g in enumerate(gb):
    ax.text(i, g + 0.7, f"{g:g} GB", ha="center", fontsize=16, fontweight="bold")
ax.set_xticks(range(4), ["DP", "ZeRO-1", "ZeRO-2", "ZeRO-3"])
ax.set_ylabel("GB per accelerator")
ax.set_ylim(0, 28)
save(fig, "zero-stages")

# 10. Precision ladder (chapter data: vendor peak dense TFLOP/s) -------------
fig, ax = plt.subplots(figsize=(7.2, 3.6))
gens = ["V100", "A100", "H100", "B200"]
fmts = [("FP32", [15.7, 19.5, 67, 75], C["gray"]),
        ("TF32", [None, 156, 494, 1125], C["blue"]),
        ("FP16/BF16", [125, 312, 989, 2250], C["orange"]),
        ("FP8", [None, None, 1979, 4500], C["brown"]),
        ("FP4", [None, None, None, 9000], C["red"])]
w = 0.16
for j, (name, vals, col) in enumerate(fmts):
    xs = [i + (j - 2) * w for i, v in enumerate(vals) if v]
    vs = [v for v in vals if v]
    ax.bar(xs, vs, w, color=col, label=name)
ax.set_yscale("log")
ax.set_ylim(8, 30000)
ax.set_xticks(range(4), gens)
ax.set_ylabel("peak TFLOP/s (log)")
ax.legend(frameon=False, ncol=5, loc="upper left", fontsize=13, handlelength=1, columnspacing=0.8)
ax.annotate("H100: 67 to 989\n(15×)", xy=(2.0, 989), xytext=(0.55, 1500), fontsize=15,
            color=C["crimson"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C["crimson"]))
save(fig, "precision-ladder")

# 11. Roofline (H100: 989 TFLOP/s, 3.35 TB/s; illustrative op placements) ----
fig, ax = plt.subplots(figsize=(6.4, 3.6))
peak, bw = 989.0, 3.35
ridge = peak / bw
xr = np.logspace(0, np.log10(3000), 200)
ax.axvspan(1, ridge, color=C["blueL"], alpha=0.5, lw=0)
ax.axvspan(ridge, 3000, color=C["orangeL"], alpha=0.5, lw=0)
ax.plot(xr[xr <= ridge], bw * xr[xr <= ridge], color=C["blue"], lw=3)
ax.plot([ridge, 3000], [peak, peak], color=C["orange"], lw=3)
ax.axvline(ridge, color=C["gray"], ls="--", lw=1.5)
ax.text(ridge * 1.08, 4, f"ridge ≈ {ridge:.0f}", fontsize=15, color=C["gray"])
pts = [("softmax", 5), ("LayerNorm", 10), ("attention\n(materialized)", 32), ("MatMul", 600)]
for name, xi in pts:
    yi = min(bw * xi, peak)
    ax.plot(xi, yi, "o", color=C["ink"], ms=9)
    ax.text(xi * 1.15, yi * 0.45, name, fontsize=14, va="top")
ax.annotate("", xy=(900, peak), xytext=(32, bw * 32 * 1.1),
            arrowprops=dict(arrowstyle="->", color=C["green"], lw=2.2))
ax.text(1.4, 750, "FlashAttention:\nfewer bytes, same FLOPs", color=C["green"], fontsize=14, fontweight="bold", va="top")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1, 3000); ax.set_ylim(1, 3000)
ax.set_xlabel("arithmetic intensity (FLOP/byte)")
ax.set_ylabel("TFLOP/s")
ax.text(1.4, 1500, "memory-bound", color=C["blue"], fontsize=15, fontweight="bold")
ax.text(330, 60, "compute-\nbound", color=C["orange"], fontsize=15, fontweight="bold", va="center")
save(fig, "training-roofline")

# 12. Communication tax: speedup = N / (1 + (N-1) r) -------------------------
fig, ax = plt.subplots(figsize=COL)
N = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
ax.plot(N, N, "--", color=C["gray"], lw=2, label="ideal")
for r, col, mk in [(0.05, C["blue"], "o"), (0.10, C["violet"], "D"), (0.40, C["red"], "^")]:
    ax.plot(N, N / (1 + (N - 1) * r), color=col, lw=2.4, marker=mk, ms=6, label=f"r = {r:.2f}")
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
ax.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
ax.set_yticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
ax.set_xlabel("accelerators N")
ax.set_ylabel("speedup")
ax.legend(frameon=False, loc="upper left", fontsize=13)
save(fig, "communication-tax")
print("ok")
