"""Slide figures for vol3 ch05 (KV Cache Management): quantitative charts.

Values are the rendered chapter's LEGO outputs (Llama 3 70B geometry on a
DGX H100 node: 327,680 bytes per token, a 526.0 GB KV pool, 512 GB/s of
aggregate host-link bandwidth, a 3,560 TFLOP/s sustained prefill rate). The
per-token payloads of the six models are the chapter figure's own inputs
(mlsysim Models.Language get_kv_cache_size at 16-bit precision).

Each figure is sized so that its slot on the 16:9 slide scales it by about
0.6, which puts 14-16 pt figure text at 8 pt or more on the slide. Column
figures are about 4.7 x 2.8 in, full-width strips about 8.6 x 1.7 in. Writes
SVG with text kept as text.

Usage: python3 make_figures.py <outdir>
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

OUT = sys.argv[1]
C = dict(blue="#006395", blueL="#D1E6F3", orange="#E67817", orangeL="#FCE4CC",
         green="#008F45", greenL="#D4EFDF", violet="#7E317B", violetL="#E6D4E5",
         red="#CB202D", redL="#F5D2D5", ink="#333333", gray="#666666",
         grid="#CCCCCC", brown="#78492A", brownL="#E3D3C8", crimson="#A51C30")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"],
    "font.size": 15, "axes.labelsize": 15, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 14, "axes.edgecolor": C["gray"], "axes.labelcolor": C["ink"],
    "xtick.color": C["ink"], "ytick.color": C["ink"], "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})

BYTES_PER_TOKEN = 327_680          # eq. 1, Llama 3 70B, 16-bit
POOL_GB = 525.994767360            # 687.2 - 141.2 - 20 GB
LINK_GBS = 512.0                   # 8 x 64 GB/s host links
SEC_PER_POS = 2 * 70.6e9 / 3.56e15  # weight work per token position


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# 1. Resident fit falls as 1/S ---------------------------------------------
fig, ax = plt.subplots(figsize=(4.7, 2.7))
s = np.logspace(np.log2(4096), np.log2(131072), 200, base=2)
ax.plot(s, POOL_GB / (s * BYTES_PER_TOKEN / 1e9), color=C["blue"], lw=2.6)
for x, y in [(8192, 195), (32768, 48), (131072, 12)]:
    ax.plot(x, y, "o", color=C["blue"], ms=9, zorder=3)
ax.annotate("195", (8192, 195), xytext=(10, 2), textcoords="offset points",
            fontsize=16, fontweight="bold", color=C["ink"])
ax.annotate("48 (r18)", (32768, 48), xytext=(8, 8), textcoords="offset points",
            fontsize=16, fontweight="bold", color=C["crimson"])
ax.annotate("12", (131072, 12), xytext=(-6, 12), textcoords="offset points",
            fontsize=16, fontweight="bold", color=C["ink"], ha="right")
ax.set_xscale("log", base=2)
ax.set_xticks([4096, 8192, 32768, 131072], ["4K", "8K", "32K", "128K"])
ax.set_xlim(3500, 160000)
ax.set_ylim(0, 290)
ax.set_yticks([0, 100, 200])
ax.set_xlabel("Retained tokens per sequence")
ax.set_ylabel("Sequences resident")
ax.grid(axis="y", color=C["grid"], lw=0.8)
save(fig, "resident-fit-curve")

# 2. Per-token KV payload by attention design --------------------------------
models = [("GPT-3 175B", 4718.592, "MHA"), ("Llama-3.1-405B", 516.096, "GQA"),
          ("Llama-3.1-70B", 327.68, "GQA"), ("Llama-2-70B", 327.68, "GQA"),
          ("Mixtral-8x7B", 131.072, "GQA"), ("DeepSeek-V3", 70.272, "MLA")]
col = {"MHA": C["gray"], "GQA": C["blue"], "MLA": C["green"]}
fig, ax = plt.subplots(figsize=(5.1, 2.9))
y = np.arange(len(models))[::-1]
for yi, (name, kb, d) in zip(y, models):
    ax.barh(yi, kb, color=col[d], height=0.7)
    ax.text(kb * 1.12, yi, f"{kb:,.0f}" if kb >= 100 else f"{kb:.1f}", va="center",
            fontsize=15, fontweight="bold" if name == "Llama-3.1-70B" else "normal")
ax.set_yticks(y, [m[0] for m in models])
ax.set_xscale("log")
ax.set_xlim(30, 40000)
ax.set_xticks([100, 1000, 10000], ["100", "1,000", "10,000"])
ax.set_xlabel("KB per token (16-bit)")
for d, c in col.items():
    ax.barh(-5, 1, color=c, label=d)
ax.set_ylim(-0.6, len(models) - 0.4)
ax.legend(frameon=False, loc="lower right", fontsize=15, handlelength=1.0,
          borderaxespad=0.1, labelspacing=0.25)
save(fig, "kv-per-token-by-design")

# 3. Paging payoff: final-length snapshots in the same pool ------------------
fig, ax = plt.subplots(figsize=(4.2, 1.65))
ax.barh([1, 0], [48, 780], color=[C["gray"], C["green"]], height=0.62)
ax.text(48 + 15, 1, "48", va="center", fontsize=17, fontweight="bold")
ax.text(780 + 15, 0, "~780", va="center", fontsize=17, fontweight="bold", color=C["green"])
ax.set_yticks([1, 0], ["full-cap", "paged"])
ax.set_xlim(0, 1000)
ax.set_xticks([0, 250, 500, 750])
ax.set_xlabel("requests in the 526 GB pool")
save(fig, "paging-payoff")

# 4. Prefix hit vs miss --------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.3, 1.65))
ax.barh([1, 0], [169, 5], color=[C["orange"], C["blue"]], height=0.62)
ax.text(169 + 4, 1, "169 ms", va="center", fontsize=16, fontweight="bold")
ax.text(5 + 4, 0, "about 5 ms", va="center", fontsize=16, fontweight="bold", color=C["blue"])
ax.set_yticks([1, 0], ["miss", "hit"])
ax.set_xlim(0, 240)
ax.set_xticks([0, 50, 100, 150])
ax.set_xlabel("ms before the new suffix is ready")
save(fig, "prefix-hit-vs-miss")

# 5. Monolithic vs chunked prefill timeline ---------------------------------
fig, axs = plt.subplots(2, 1, figsize=(8.2, 2.45), sharex=True)
mono = 4096 * SEC_PER_POS * 1e3
ax = axs[0]
ax.barh(0, mono, left=0, color=C["orange"], height=0.6)
ax.text(mono / 2, 0, "4,096-token prefill, one unit", ha="center", va="center",
        color="white", fontsize=15, fontweight="bold")
ax.barh(0, 5, left=mono, color=C["blue"], height=0.6)
for t in (0, mono + 5):
    ax.plot(t, 0.62, marker="v", color=C["blue"], ms=11, clip_on=False)
ax.text(mono + 12, 0, "gap: about 168 ms", va="center", fontsize=15,
        color=C["red"], fontweight="bold")
ax.text(0, 1.05, "(a) monolithic", fontsize=15, fontweight="bold", va="bottom")
ax2 = axs[1]
it = 544 * SEC_PER_POS * 1e3
for k in range(8):
    ax2.barh(0, it, left=k * it, color=C["orange"], edgecolor="white", height=0.6)
    ax2.text(k * it + it / 2, 0, str(k + 1), ha="center", va="center", color="white",
             fontsize=15, fontweight="bold")
    ax2.plot(k * it, 0.62, marker="v", color=C["blue"], ms=11, clip_on=False)
ax2.plot(8 * it, 0.62, marker="v", color=C["blue"], ms=11, clip_on=False)
ax2.text(8 * it + 7, 0, "done at about 173 ms", va="center", fontsize=15)
ax2.text(0, 1.05, "(b) 512-token chunks: gap about 22 ms", fontsize=15, fontweight="bold",
         va="bottom")
for a in axs:
    a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.set_ylim(-0.4, 1.0)
axs[1].set_xlim(0, 265)
axs[1].set_xticks([0, 50, 100, 150, 200])
axs[1].set_xlabel("ms; blue marks are output tokens of the 32 ongoing streams")
fig.subplots_adjust(hspace=0.75)
save(fig, "chunked-timeline")

# 6. Reload vs recompute -----------------------------------------------------
reload_s = 32768 * BYTES_PER_TOKEN / (LINK_GBS * 1e9)
recompute_s = 2 * 70.6e9 * 32768 / 3.56e15
fig, ax = plt.subplots(figsize=(7.0, 1.6))
bars = [("reload", reload_s, C["blue"]),
        ("reload, 1/4 link", 4 * reload_s, C["blueL"]),
        ("recompute", recompute_s, C["orange"])]
for i, (lab, v, c) in enumerate(bars):
    ax.barh(2 - i, v, color=c, height=0.66, edgecolor=C["blue"] if c == C["blueL"] else c)
txt = ["21 ms", "84 ms, still 15× under recompute", "1.3 s = 62× the reload"]
for i, t in enumerate(txt):
    ax.text(bars[i][1] * 1.15, 2 - i, t, va="center", fontsize=15, fontweight="bold")
ax.set_yticks([2, 1, 0], [b[0] for b in bars])
ax.set_xscale("log")
ax.set_xlim(0.005, 80)
ax.set_xticks([0.01, 0.1, 1, 10], ["10 ms", "100 ms", "1 s", "10 s"])
ax.set_xlabel("time to restore r18's 10.74 GB")
save(fig, "reload-vs-recompute")

# 7. Keep / offload / park regimes ----------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 2.9))
S = np.logspace(np.log2(2048), np.log2(131072), 100, base=2)
rt = 2 * S * BYTES_PER_TOKEN / (LINK_GBS * 1e9)
host = 60.0
ax.fill_betweenx(S, 1e-3, rt, color=C["greenL"])
ax.fill_betweenx(S, rt, host, color=C["violetL"])
ax.fill_betweenx(S, host, 3600, color="#EEEEEE")
ax.plot(rt, S, color=C["green"], lw=2.4)
ax.axvline(host, color=C["gray"], lw=2, ls="--")
ax.text(1.4e-3, 60000, "keep", fontsize=16, fontweight="bold", color=C["green"])
ax.text(0.5, 60000, "move", fontsize=16, fontweight="bold", color=C["violet"])
ax.text(110, 45000, "park\nor\ndrop", fontsize=15, fontweight="bold", color=C["ink"], va="center")
ax.text(0.012, 2600, "round trip", fontsize=15, color=C["green"], fontweight="bold")
ax.plot(45, 32768, "o", color=C["crimson"], ms=10, zorder=4)
ax.plot([rt[np.argmin(abs(S - 32768))], 45], [32768, 32768], ":", color=C["crimson"], lw=2)
ax.text(40, 18000, "r18: 45 s", fontsize=15, color=C["crimson"], ha="right", va="top",
        fontweight="bold")
ax.set_xscale("log")
ax.set_yscale("log", base=2)
ax.set_xlim(1e-3, 3600)
ax.set_ylim(2048, 131072)
ax.set_xticks([1e-3, 0.1, 10, 3600], ["1 ms", "0.1 s", "10 s", "1 h"])
ax.set_xticks([1e-2, 1, 60], minor=True)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks([2048, 8192, 32768, 131072], ["2K", "8K", "32K", "128K"])
ax.set_xlabel("expected tool wait (dashed: host fills)")
ax.set_ylabel("context tokens")
save(fig, "pause-regimes")

# 8. Correlated return burst ---------------------------------------------------
fig, ax = plt.subplots(figsize=(4.7, 2.7))
k = np.arange(1, 33)
ax.bar(k, k * reload_s * 1e3, color=C["blue"], width=0.8)
ax.axhline(reload_s * 1e3, color=C["green"], lw=2, ls="--")
ax.text(0.5, 470, "dashed: one\nreload alone,\n21 ms", fontsize=15, color=C["green"],
        fontweight="bold", va="center")
ax.text(32, 700, "last: 0.67 s", ha="right", fontsize=16, color=C["red"], fontweight="bold")
ax.set_xlabel("paused repair, reload order")
ax.set_ylabel("reload done (ms)")
ax.set_xticks([1, 8, 16, 24, 32])
ax.set_yticks([0, 200, 400, 600])
ax.set_ylim(0, 790)
save(fig, "return-burst")

# 9. Sharing gain 1/(1 - f) ----------------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 2.8))
f = np.linspace(0, 0.9, 200)
ax.plot(f * 100, 1 / (1 - f), color=C["green"], lw=2.6)
for x in (18.8, 70.6, 75.0):
    ax.plot(x, 1 / (1 - x / 100), "o", color=C["blue"], ms=9, zorder=3)
ax.text(22, 0.3, "repair prompt: 1.25×", fontsize=15, color=C["ink"])
ax.annotate("repair fleet,\nshared\nworkspace: 3.4×", (70.6, 3.4), xytext=(14, 4.4),
            fontsize=15, color=C["ink"],
            arrowprops=dict(arrowstyle="-", color=C["gray"], lw=1.2))
ax.text(79, 3.3, "support:\n4×", fontsize=15, color=C["ink"], va="top")
ax.set_xlim(0, 95)
ax.set_ylim(0, 10.5)
ax.set_yticks([1, 4, 7, 10])
ax.set_xlabel("shared fraction (%)")
ax.set_ylabel("fit gain")
ax.grid(axis="y", color=C["grid"], lw=0.8)
save(fig, "sharing-gain")

# 10. Residency tax ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.0, 1.55))
for row in (1, 0):
    ax.barh(row, 5, color=C["orange"], height=0.62)
    ax.text(2.5, row, "gen", ha="center", va="center", color="white", fontsize=14,
            fontweight="bold")
ax.barh(1, 45, left=5, color="#BBBBBB", height=0.62)
ax.text(27.5, 1, "tool wait: slot held, no tokens", ha="center", va="center",
        fontsize=15, color=C["ink"])
ax.text(6.5, 0, "slot released during the wait", va="center", fontsize=15, color=C["gray"])
ax.text(51.5, 1, "41/50 s = 0.82 turns/s", va="center", fontsize=15, fontweight="bold",
        color=C["red"])
ax.text(51.5, 0, "41/5 s = 8.2 turns/s", va="center", fontsize=15, fontweight="bold",
        color=C["green"])
ax.set_yticks([1, 0], ["keep", "release"])
ax.set_xlim(0, 77)
ax.set_xticks([0, 5, 50])
ax.set_xlabel("seconds one turn holds its KV slot")
ax.set_ylim(-0.5, 1.5)
save(fig, "residency-tax")
print("ok")
