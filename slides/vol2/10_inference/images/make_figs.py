#!/usr/bin/env python3
"""Figures for Volume II, Lecture 10 (Serving at Scale).

Every value is sourced. "Ch." is the rendered chapter, Serving at Scale
(Volume II, Chapter 10). Values computed here are labeled "derived" on the
figure or on its slide; scenario inputs are labeled "illustrative".

Sources
  - Ch. Napkin Math 10.1: training $2M; 10^6 users x 50 requests/day x 365
    = 18.25B requests/year at an assumed $0.001 each = $18.25M; ratio 9.1.
  - Ch. Sec. 10.3.2: 100 ms target, 70 ms model execution, 30 ms for the
    rest; four sequential exchanges of 50 to 500 us = 0.2 to 2 ms.
  - InferenceX (SemiAnalysis, Apache-2.0), DeepSeek-R1-0528, FP8, 1,024 in /
    1,024 out, snapshot 2026-10-12, as stored in MLSysIM
    (mlsysim/literature/data/series/inferencex_dsr1_fp8_1k1k.csv). Frontier =
    configurations no other configuration on the same accelerator beats on
    both axes (the chapter's rule, Sec. 10.6.2).
  - Ch. Napkin Math 10.2: knee t0/treq; vision 2 ms / 1 ms = 2; LLM decode
    40 ms / 0.5 ms = 80.
  - Ch. Napkin Math 10.3: outputs 50, 200, 100, 150 tokens; 500 of 800
    sequence-iterations useful; waste 37.5 percent.
  - Azure LLM inference trace 2023 (Patel et al., Splitwise, ISCA 2024;
    CC BY): conversation and code samples, one hour, 16 Nov 2023. Waste of
    fixed batches of 8 consecutive requests derived here.
  - Azure LLM inference trace 2024, conversation, 12 to 18 May 2024
    (Stojkovic et al., DynamoLLM, HPCA 2025; CC BY), 10-minute bins made by
    data/aggregate_azure_2024.py.
  - Ch. Napkin Math 10.5: 327,680 B per token (80 layers, 8 KV heads of 128,
    FP16); 42.9 GB per 131,072-token request; 526.0 GB free; 12 requests.
  - Ch. Sec. 10.9 and Napkin Math 10.7: 70B FP16 140 GB, INT4 35 GB, 175B
    INT4 87.5 GB; GQA maximum batch 372 (FP16) and 901 (INT4 weights, INT8
    KV), 2.4x.
  - Ch. Sec. 10.8.1: prefill intensity about prompt length, decode about
    batch; H100 FP16 ridge 989 TFLOP/s / 3.35 TB/s = 295 FLOP/byte (derived).
  - Ch. Example 10.5: 10,000-token prompt, 512-token chunks, 20 iterations.
  - Ch. Example 10.6: 30 ms serial layer (12 attention + 18 FFN), TP 8,
    0.3 ms per AllReduce, 4.35 ms per layer, 2,400 to 348 ms, 6.9x.
  - Ch. Sec. 10.15.1, table 26: decode step 85 / 69.7 / 49.6 ms; speculation
    K = 5, 4 ms draft, acceptance 0.78, 3.52 tokens, 53.6 ms round,
    15.2 ms per token.
  - Ch. Sec. 10.11.2, equations 14 and 15 (Mitzenmacher 2001): static balls
    into bins. Simulated here.
  - Ch. Sec. 10.11.1: P(slow) = 1 - (1 - q)^N; 100 leaves at p99 -> 63%.
  - Ch. Example 10.11: cold start 6.7 min.
  - Ch. table 25 (hypothetical price snapshot): H100 on-demand $4.00/hour.

Run from this folder: python3 make_figs.py   (then make deck V=2 D=10_inference)
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

C = sp.C
plt = sp.plt
out = Path(__file__).parent
DATA = out / "data"
MLSYSIM = Path(__file__).resolve().parents[4] / "mlsysim"
IX = MLSYSIM / "mlsysim" / "literature" / "data" / "series" / "inferencex_dsr1_fp8_1k1k.csv"


def hide(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def note(ax, text, x=0.0, y=-0.28, ha="left"):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9, color=C["neutral"],
            ha=ha, va="top")


# =============================================================== hook: cost
def hook(k):
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    rows = [("Training", 2.0, "neutral", "\\$2.0M, once"),
            ("Serving,\nfirst year", 18.25, "compute", "\\$18.25M  (9.1×)")]
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = 1 - i
        shown = i == 0 or k >= 3
        ax.barh(y, v, color=C[ck], height=0.55, alpha=1.0 if shown else 0.0)
        ax.text(-0.3, y, lab, ha="right", va="center", fontsize=13, fontweight="bold",
                alpha=1.0 if (i == 0 or k >= 2) else 0.0)
        ax.text(v + 0.3, y, txt, va="center", fontsize=13, fontweight="bold",
                color=C[ck], alpha=1.0 if shown else 0.0)
        if i == 1 and k == 2:
            ax.text(0.4, y, "???", va="center", fontsize=15, fontweight="bold",
                    color=C["danger"])
    ax.set_xlim(0, 25)
    ax.set_ylim(-0.6, 1.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Million dollars, one 70B model")
    sp.save(fig, out / f"hook_step{k}.svg")


for k in (1, 2, 3):
    hook(k)

# ============================================== request lifecycle (TTFT/TPOT)
def lifecycle(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    y = 0.0
    # host stages then prefill then decode ticks then stream
    ax.add_patch(Rectangle((0, y), 1.6, 0.6, color=C["neutral_l"]))
    ax.text(0.8, y + 0.3, "route,\nqueue", ha="center", va="center", fontsize=11)
    ax.add_patch(Rectangle((1.6, y), 1.8, 0.6, color=C["compute"]))
    ax.text(2.5, y + 0.3, "prefill", ha="center", va="center", fontsize=12,
            color="white", fontweight="bold")
    ticks = [3.4 + 0.55 * i for i in range(12)]
    if k >= 3:
        ticks = ticks[:6] + [t + 1.3 for t in ticks[6:]]
    for t in ticks:
        ax.add_patch(Rectangle((t, y), 0.38, 0.6, color=C["memory"]))
    ax.text(5.4, y + 0.85, "decode: one token per step", ha="center", fontsize=11,
            color=C["memory"], fontweight="bold")
    ax.annotate("", xy=(3.4, -0.35), xytext=(0, -0.35),
                arrowprops=dict(arrowstyle="<->", color=C["text"], lw=1.4))
    ax.text(1.7, -0.62, "TTFT", ha="center", fontsize=11,
            fontweight="bold")
    if k >= 2:
        a, b = ticks[0], ticks[-1] + 0.38
        ax.annotate("", xy=(b, -0.35), xytext=(a, -0.35),
                    arrowprops=dict(arrowstyle="<->", color=C["memory"], lw=1.4))
        ax.text((a + b) / 2, -0.62, "TPOT: mean gap between tokens", ha="center", fontsize=11,
                color=C["memory"], fontweight="bold")
    if k >= 3:
        a, b = ticks[5] + 0.38, ticks[6]
        ax.add_patch(Rectangle((a, y), b - a, 0.6, color=C["danger_l"]))
        ax.text((a + b) / 2, y + 0.3, "stall", ha="center", va="center", fontsize=11,
                color=C["danger"], fontweight="bold")
        ax.text((a + b) / 2, 1.2, "one long gap (TBT tail)\ncan hide inside a good mean", ha="center",
                fontsize=10.5, color=C["danger"], fontweight="bold")
    ax.set_xlim(-0.1, 11.0)
    ax.set_ylim(-0.85, 1.6)
    hide(ax)
    sp.save(fig, out / f"lifecycle_step{k}.svg")


for k in (1, 2, 3):
    lifecycle(k)

# ======================================================= serving tax budget
def tax(k):
    fig, ax = plt.subplots(figsize=(5.8, 1.7))
    ax.barh(0, 70, color=C["compute"], height=0.6)
    ax.text(35, 0, "model execution  70 ms", ha="center", va="center", color="white",
            fontsize=12, fontweight="bold")
    ax.barh(0, 30, left=70, color=C["neutral_l"], height=0.6)
    ax.text(85, 0, "30 ms", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.axvline(100, color=C["danger"], ls="--", lw=2)
    ax.text(100, 0.42, "100 ms target ", color=C["danger"], fontsize=11, fontweight="bold",
            va="bottom", ha="right")
    if k >= 2:
        ax.barh(0, 2, left=70, color=C["network"], height=0.6)
        ax.annotate("4 exchanges: 0.2 to 2 ms", (71, -0.3), xytext=(40, -0.95),
                    color=C["network"], fontsize=11, fontweight="bold", va="center", ha="center",
                    arrowprops=dict(arrowstyle="-", color=C["network"], lw=1.2))
        ax.text(80, -0.95, "routing, serialization,\ncoordination, queueing", color=C["neutral"],
                fontsize=10, fontweight="bold", va="center", ha="center")
    ax.set_xlim(0, 104)
    ax.set_ylim(-1.45, 0.85)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Milliseconds of one request's budget (illustrative)")
    sp.save(fig, out / f"tax_step{k}.svg")


for k in (1, 2):
    tax(k)

# ===================================================== InferenceX frontiers
ix = list(csv.DictReader(open(IX)))
for x in ix:
    for f in ("conc", "disagg", "interactivity_tok_s_user", "output_tok_s_gpu"):
        x[f] = float(x[f])


def frontier(rows):
    pts = sorted(((r["interactivity_tok_s_user"], r["output_tok_s_gpu"]) for r in rows),
                 key=lambda p: -p[0])
    f, best = [], -1.0
    for a, b in pts:
        if b > best:
            f.append((a, b))
            best = b
    return sorted(f)


def best_at(rows, t):
    return max(r["output_tok_s_gpu"] for r in rows if r["interactivity_tok_s_user"] >= t)


HW = [("h100", "H100", "neutral"), ("h200", "H200", "memory"), ("mi355x", "MI355X", "data"),
      ("b200", "B200", "compute"), ("gb200", "GB200 NVL72", "network")]
b200 = [r for r in ix if r["hardware"] == "b200"]
fb = frontier(b200)
b200_peak = max(fb, key=lambda p: p[1])
b200_100 = best_at(b200, 100)
h100_50 = best_at([r for r in ix if r["hardware"] == "h100"], 50)
gb200_50 = best_at([r for r in ix if r["hardware"] == "gb200"], 50)
print(f"InferenceX: {len(ix)} configs; B200 peak {b200_peak}; B200 at >=100: {b200_100:.0f}; "
      f"span {b200_peak[1] / b200_100:.1f}x; H100@50 {h100_50:.1f}; GB200@50 {gb200_50:.0f}; "
      f"ratio {gb200_50 / h100_50:.1f}")


LABX = {"h100": 75, "h200": 150, "mi355x": 185, "b200": 225, "gb200": 95}
LABDY = {"h100": 0.4, "h200": 0.4, "mi355x": 0.4, "b200": 2.0, "gb200": 2.0}


def ixplot(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    shown = ["b200"] if k == 1 else [h for h, _, _ in HW]
    for h, lab, ck in HW:
        rows = [r for r in ix if r["hardware"] == h]
        f = frontier(rows)
        vis = h in shown
        ax.scatter([r["interactivity_tok_s_user"] for r in rows],
                   [r["output_tok_s_gpu"] for r in rows], s=10, color=C[ck],
                   alpha=0.25 if vis else 0.0, lw=0)
        ax.plot([p[0] for p in f], [p[1] for p in f], color=C[ck], lw=2.4,
                alpha=1.0 if vis else 0.0)
        lx = LABX[h]
        ly = max([p[1] for p in f if p[0] >= lx] or [f[-1][1]])
        ax.text(lx, ly * LABDY[h], lab, ha="left", va="center", fontsize=11, color=C[ck],
                fontweight="bold", alpha=1.0 if vis else 0.0)
    if k == 1:
        ax.plot(*b200_peak, "o", ms=9, color=C["compute"], mec="white")
        ax.annotate(f"{b200_peak[1]:,.0f} at {b200_peak[0]:.0f} tok/s per user",
                    b200_peak, xytext=(25, 2), textcoords="offset points", fontsize=11,
                    fontweight="bold", color=C["compute"])
        x100 = [r for r in b200 if r["output_tok_s_gpu"] == b200_100][0]["interactivity_tok_s_user"]
        ax.plot(x100, b200_100, "o", ms=9, color=C["compute"], mec="white")
        ax.annotate(f"{b200_100:,.0f} at 100+: 30× less", (x100, b200_100), xytext=(-20, -38),
                    textcoords="offset points", fontsize=11, fontweight="bold",
                    color=C["danger"])
    if k >= 3:
        ax.axvline(50, color=C["text"], ls="--", lw=1.2)
        for v, ck in ((h100_50, "neutral"), (gb200_50, "network")):
            ax.plot(50, v, "o", ms=8, color=C[ck], mec="white", zorder=5)
            ax.text(46, v, f"{v:,.0f}", ha="right", va="center", fontsize=11, fontweight="bold",
                    color=C[ck], bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.set_yscale("log")
    ax.set_ylim(5, 15000)
    ax.set_xlim(-30, 280)
    ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.set_xlabel("Output tokens/s each user receives (1 / TPOT)")
    ax.set_ylabel("Output tokens/s per GPU")
    sp.save(fig, out / f"frontier_step{k}.svg")


for k in (1, 2, 3):
    ixplot(k)

# ============================================================ batching knee
def knee():
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    B = np.logspace(0, 3, 300)
    for lab, kn, ck, xy in (("vision: knee at\n2 ms / 1 ms = 2", 2, "data", (2.6, 0.2)),
                            ("LLM decode: knee at\n40 ms / 0.5 ms = 80", 80, "memory", (120, 0.2))):
        y = B / (kn + B)
        ax.plot(B, y, color=C[ck], lw=3)
        ax.plot([kn], [0.5], "o", ms=10, color=C[ck], mec="white", mew=1.5)
        ax.text(xy[0], xy[1], lab, fontsize=11, fontweight="bold", color=C[ck], va="center")
    ax.axhline(1.0, color=C["neutral"], ls="--", lw=1.2)
    ax.text(1.05, 1.03, "asymptote 1 / t_req", fontsize=10.5, color=C["neutral"])
    ax.set_xscale("log")
    ax.set_ylim(0, 1.15)
    ax.set_xlim(1, 1000)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.set_xlabel("Batch size B")
    ax.set_ylabel("Share of top rate")
    ax.set_yticks([0, 0.5, 1.0])
    sp.save(fig, out / "knee.svg")


knee()

# ============================ measured: concurrency on one colocated B200 setup
def conc():
    rows = sorted([r for r in ix if r["hardware"] == "b200" and r["framework"] == "sglang"],
                  key=lambda r: r["conc"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.8, 2.7), sharex=True)
    xs = [r["conc"] for r in rows]
    a1.scatter(xs, [r["output_tok_s_gpu"] for r in rows], s=28, color=C["compute"], zorder=3)
    a2.scatter(xs, [r["interactivity_tok_s_user"] for r in rows], s=28, color=C["memory"], zorder=3)
    a1.set_ylabel("tok/s per GPU", fontsize=11)
    a2.set_ylabel("tok/s per user", fontsize=11)
    a1.text(1, 900, "17 to 1,073 per GPU (63×)", fontsize=11, fontweight="bold", color=C["compute"])
    a2.text(40, 150, "139 to 18 per user (7.8× slower)", fontsize=11, fontweight="bold",
            color=C["memory"])
    a1.set_ylim(0, 1300)
    a2.set_ylim(0, 260)
    a2.set_xscale("log", base=2)
    a2.set_xticks([1, 4, 16, 64, 256, 512])
    a2.set_xticklabels(["1", "4", "16", "64", "256", "512"])
    a2.set_xlabel("Concurrent requests (8 B200 GPUs, SGLang, colocated)")
    sp.save(fig, out / "conc_measured.svg")


conc()

# =================================== Azure trace: output lengths and slot waste
def load_trace(name):
    r = list(csv.DictReader(open(DATA / name)))
    t = [datetime.strptime(x["TIMESTAMP"][:26], "%Y-%m-%d %H:%M:%S.%f") for x in r]
    c = np.array([int(x["ContextTokens"]) for x in r])
    g = np.array([int(x["GeneratedTokens"]) for x in r])
    return t, c, g


def waste(g, B):
    n = len(g) // B
    gb = g[: n * B].reshape(n, B)
    return float((1 - gb.mean(1) / gb.max(1)).mean())


tv, cv, gv = load_trace("azure_llm_conv_2023.csv")
tc, cc, gc = load_trace("azure_llm_code_2023.csv")
w_conv, w_code = waste(gv, 8), waste(gc, 8)
print(f"Azure conv: n={len(gv)}, prompt median {np.median(cv):.0f}, output median "
      f"{np.median(gv):.0f} mean {gv.mean():.0f} p99 {np.percentile(gv, 99):.0f} max {gv.max()}; "
      f"waste B=8 {w_conv:.3f}; code waste {w_code:.3f}; code prompt median {np.median(cc):.0f} "
      f"output median {np.median(gc):.0f}")


def lengths():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.4), gridspec_kw={"width_ratios": [1.5, 1]})
    a1.hist(gv, bins=np.arange(0, 1001, 25), color=C["memory"])
    for v, lab, dy in ((np.median(gv), "median 129", 0.92), (gv.mean(), "mean 211", 0.70)):
        a1.axvline(v, color=C["text"], lw=1.2, ls="--")
        a1.text(v + 15, a1.get_ylim()[1] * dy, lab, fontsize=10.5, fontweight="bold")
    a1.set_xlabel("Output tokens per chat request")
    a1.set_ylabel("Requests")
    a1.set_xlim(0, 1000)
    bars = [("Ch. example", 37.5, "neutral"), ("Chat trace", 100 * w_conv, "memory"),
            ("Code trace", 100 * w_code, "compute")]
    for i, (lab, v, ck) in enumerate(bars):
        a2.barh(2 - i, v, color=C[ck], height=0.6)
        a2.text(v + 2, 2 - i, f"{v:.1f}%" if v == 37.5 else f"{v:.0f}%", va="center", fontsize=12, fontweight="bold", color=C[ck])
        a2.text(-3, 2 - i, lab, ha="right", va="center", fontsize=11, fontweight="bold")
    a2.set_xlim(0, 90)
    a2.set_yticks([])
    a2.spines["left"].set_visible(False)
    a2.set_xlabel("Slot time idle, batch of 8")
    sp.save(fig, out / "lengths.svg")


lengths()

# ===================================== continuous batching schedule (Napkin 10.3)
OUTS = [50, 200, 100, 150]


def cb(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    names = ["R1", "R2", "R3", "R4"]
    nxt = {0: "R5", 2: "R6", 3: "R7"}
    for s, L in enumerate(OUTS):
        y = 3 - s
        ax.add_patch(Rectangle((0, y), L, 0.7, color=C["memory"]))
        ax.text(L / 2, y + 0.35, f"{names[s]}  {L}", ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")
        if L < 200:
            if k == 1:
                ax.add_patch(Rectangle((L, y), 200 - L, 0.7, facecolor="white",
                                       edgecolor=C["danger"], hatch="///", lw=1.0))
            else:
                ax.add_patch(Rectangle((L, y), 200 - L, 0.7, color=C["data"]))
                ax.text(L + (200 - L) / 2, y + 0.35, nxt[s] if L > 120 else f"{nxt[s]} joins at {L}", ha="center",
                        va="center", color="white", fontsize=11, fontweight="bold")
    if k == 1:
        ax.text(205, 1.9, "idle: 300 of 800\nslot-iterations\n= 37.5%", fontsize=11,
                color=C["danger"], fontweight="bold", va="center")
    else:
        ax.text(205, 1.9, "7 requests\nadvance in the\nsame 200\niterations", fontsize=11,
                color=C["data"], fontweight="bold", va="center")
    ax.set_xlim(0, 260)
    ax.set_ylim(-0.1, 3.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlabel("Decode iteration (20 ms each, illustrative)")
    sp.save(fig, out / f"cb_step{k}.svg")


for k in (1, 2):
    cb(k)

# ============================================ KV admission: 12 versus 1,397
KV_TOKEN = 2 * 80 * 8 * 128 * 2          # bytes per token (Napkin Math 10.5)
FREE = 526.0e9
LONG = 131072 * KV_TOKEN
CHAT_TOK = int(np.median(cv) + np.median(gv))
CHAT = CHAT_TOK * KV_TOKEN
print(f"KV: {KV_TOKEN} B/token; 128K request {LONG / 1e9:.1f} GB -> {int(FREE // LONG)}; "
      f"median chat {CHAT_TOK} tokens {CHAT / 1e6:.1f} MB -> {int(FREE // CHAT)}")


def kv(k):
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    W = 1.5
    ax.add_patch(Rectangle((0, 0), W, 526, fill=False, ec=C["text"], lw=1.2))
    for i in range(12):
        ax.add_patch(Rectangle((0.06, i * 42.9 + 1.5), W - 0.12, 40.5, color=C["memory"]))
    ax.text(W / 2, 260, "12 requests\n× 42.9 GB\n(128K tokens)", ha="center", va="center",
            color="white", fontsize=12, fontweight="bold",
            bbox=dict(fc=C["memory"], ec="none", pad=2))
    ax.plot([0, 2 * W + 0.6], [526, 526], color=C["danger"], lw=2)
    ax.text(W + 0.3, 545, "526 GB free for KV, 8× H100", ha="center", fontsize=11.5,
            color=C["danger"], fontweight="bold")
    a = 1.0 if k >= 2 else 0.0
    x2 = W + 0.6
    ax.add_patch(Rectangle((x2, 0), W, 526, fill=False, ec=C["text"], lw=1.2, alpha=a))
    ax.add_patch(Rectangle((x2 + 0.06, 1.5), W - 0.12, 523, color=C["data"], alpha=a))
    ax.text(x2 + W / 2, 260, f"{int(FREE // CHAT):,} requests\n× {CHAT / 1e6:.0f} MB\n(median chat,\n{CHAT_TOK:,} tokens)",
            ha="center", va="center", color="white", fontsize=12, fontweight="bold", alpha=a)
    ax.set_xlim(-0.05, 2 * W + 0.65)
    ax.set_ylim(0, 580)
    hide(ax)
    sp.save(fig, out / f"kv_step{k}.svg")


for k in (1, 2):
    kv(k)

# ======================================================= paging diagram
def paging(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    # top row: contiguous reservation, 16 slots
    ax.text(-0.3, 1.35, "Reserve\nmaximum", ha="right", va="center", fontsize=11, fontweight="bold")
    used = {0: "A", 1: "A", 2: "A", 8: "B", 9: "B", 10: "B", 11: "B", 12: "B"}
    for i in range(16):
        reserved = i < 16
        lab = used.get(i, "")
        col = C["memory"] if lab else "white"
        ax.add_patch(Rectangle((i, 1.0), 0.92, 0.7, facecolor=col, edgecolor=C["memory"],
                               hatch="" if lab else "///", lw=1.0))
        if lab:
            ax.text(i + 0.46, 1.35, lab, ha="center", va="center", color="white", fontsize=11,
                    fontweight="bold")
    ax.text(16.2, 1.35, "8 of 16 slots\nhold tokens", va="center", fontsize=10.5,
            color=C["danger"], fontweight="bold")
    if k >= 2:
        ax.text(-0.3, -0.05, "Pages of\n4 slots", ha="right", va="center", fontsize=11,
                fontweight="bold")
        layout = ["A", "B", "A", "B", "C", "", "C", "D"]
        fill = {"A": 3, "B": 5, "C": 6, "D": 2}
        seen = {}
        for b, who in enumerate(layout):
            x0 = b * 2.0
            if who:
                seen[who] = seen.get(who, 0) + 1
                n = min(4, fill[who] - 4 * (seen[who] - 1))
            else:
                n = 0
            for j in range(4):
                filled = j < n
                ax.add_patch(Rectangle((x0 + j * 0.45, -0.4), 0.42, 0.7,
                                       facecolor=C["data"] if filled else "white",
                                       edgecolor=C["data"], lw=1.0))
            if who:
                ax.text(x0 + 0.9, 0.42, who, ha="center", fontsize=10.5, fontweight="bold",
                        color=C["data"])
        ax.text(16.2, -0.05, "waste ≤ one\npartly filled page", va="center", fontsize=10.5,
                color=C["data"], fontweight="bold")
    ax.set_xlim(-3.2, 20.5)
    ax.set_ylim(-0.6, 2.0)
    hide(ax)
    sp.save(fig, out / f"paging_step{k}.svg")


for k in (1, 2):
    paging(k)

# ============================================================== precision
def precision():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.3), gridspec_kw={"width_ratios": [1.35, 1],
                                                                          "wspace": 0.75})
    rows = [("70B FP16", 140, "memory"), ("70B INT4", 35, "data"), ("175B INT4", 87.5, "compute")]
    for i, (lab, v, ck) in enumerate(rows):
        a1.barh(2 - i, v, color=C[ck], height=0.6)
        a1.text(max(v + 3, 84), 2 - i, f"{v:g} GB", va="center", fontsize=11.5, fontweight="bold", color=C[ck])
        a1.text(-4, 2 - i, lab, ha="right", va="center", fontsize=11, fontweight="bold")
    a1.axvline(80, color=C["danger"], ls="--", lw=1.8)
    a1.text(82, 2.45, "one 80 GB GPU", color=C["danger"], fontsize=10, fontweight="bold")
    a1.set_xlim(0, 230)
    a1.set_ylim(-0.6, 2.8)
    a1.set_yticks([])
    a1.spines["left"].set_visible(False)
    a1.set_xlabel("Weight bytes")
    for i, (lab, v, ck) in enumerate((("FP16", 372, "neutral"), ("INT4 W,\nINT8 KV", 901, "data"))):
        a2.barh(1 - i, v, color=C[ck], height=0.6)
        a2.text(v + 15, 1 - i, f"{v}", va="center", fontsize=12, fontweight="bold", color=C[ck])
        a2.text(-15, 1 - i, lab, ha="right", va="center", fontsize=11, fontweight="bold")
    a2.text(380, -0.75, "2.4×", fontsize=13, fontweight="bold", color=C["data"])
    a2.set_xlim(0, 1150)
    a2.set_ylim(-1.0, 1.6)
    a2.set_yticks([])
    a2.spines["left"].set_visible(False)
    a2.set_xlabel("Max batch, 70B GQA, 8× H100")
    sp.save(fig, out / "precision.svg")


precision()

# =============================================================== roofline
def roofline(k):
    fig, ax = plt.subplots(figsize=(4.3, 2.7))
    R = 295.0
    xs = np.logspace(-0.3, 4, 300)
    roof = np.minimum(1.0, xs / R)
    ax.plot(xs[xs <= R], roof[xs <= R], color=C["memory"], lw=3)
    ax.plot(xs[xs >= R], roof[xs >= R], color=C["compute"], lw=3)
    ax.fill_between(xs[xs <= R], 1e-3, roof[xs <= R], color=C["memory_l"], alpha=0.6)
    ax.fill_between(xs[xs >= R], 1e-3, roof[xs >= R], color=C["compute_l"], alpha=0.6)
    ax.axvline(R, color=C["neutral"], lw=1, ls="--")
    ax.text(R * 1.1, 0.0045, "ridge 295\n(H100 FP16,\nderived)", color=C["neutral"], fontsize=10)
    pts = [("decode, B = 1", 1.0, "memory", (10, -16), "left"),
           ("decode, B = 80", 80.0, "memory", (-10, 8), "right"),
           ("prefill,\n1,024-token\nprompt", 1024.0, "compute", (0, -48), "center")]
    for i, (lab, x, ck, off, ha) in enumerate(pts[:k]):
        y = min(1.0, x / R)
        ax.plot([x], [y], "o", ms=11, color=C[ck], mec="white", mew=1.5, zorder=5)
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", ha=ha,
                    fontsize=11.5, fontweight="bold", color=C[ck])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 1e4)
    ax.set_ylim(1e-3, 3)
    ax.minorticks_off()
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.set_yticks([])
    ax.set_xlabel("FLOP per byte of FP16 weights")
    ax.set_ylabel("Attained rate (schematic)")
    sp.save(fig, out / f"roofline_step{k}.svg")


for k in (1, 2, 3):
    roofline(k)

# ========================================================= chunked prefill
def chunk(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    def row(y, label, segs):
        ax.text(-0.3, y + 0.3, label, ha="right", va="center", fontsize=11, fontweight="bold")
        for x0, w, ck in segs:
            ax.add_patch(Rectangle((x0, y), w, 0.6, color=C[ck], ec="white", lw=0.8))
    # monolithic: decode, decode, big prefill, decode...
    segs = [(0, 1, "memory"), (1, 1, "memory"), (2, 8, "compute")] + \
           [(10 + i, 1, "memory") for i in range(6)]
    row(1.3, "Whole\nprompt", segs)
    ax.text(6, 1.6, "10,000-token prefill", ha="center", va="center", color="white",
            fontsize=11, fontweight="bold")
    ax.annotate("", xy=(10, 2.15), xytext=(2, 2.15),
                arrowprops=dict(arrowstyle="<->", color=C["danger"], lw=1.4))
    ax.text(6, 2.3, "every active stream waits", ha="center", fontsize=10.5, color=C["danger"],
            fontweight="bold")
    if k >= 2:
        segs = [(0, 1, "memory"), (1, 1, "memory")]
        x = 2
        for i in range(10):
            segs.append((x, 0.45, "memory"))
            segs.append((x + 0.45, 0.95, "compute"))
            x += 1.4
        row(0.0, "512-token\nchunks", segs)
        ax.text(8.5, -0.35, "each iteration: decode tokens + one chunk  (≥ 20 iterations)",
                ha="center", fontsize=10.5, color=C["text"], fontweight="bold")
        ax.text(16.2, 0.3, "gap ≤ one\niteration;\nfirst token\nlater", fontsize=10.5,
                va="center", color=C["compute"], fontweight="bold")
    ax.set_xlim(-2.6, 19.5)
    ax.set_ylim(-0.6, 2.6)
    hide(ax)
    sp.save(fig, out / f"chunk_step{k}.svg")


for k in (1, 2):
    chunk(k)

# ============================================== disaggregated vs colocated
def disagg():
    fig, axs = plt.subplots(1, 2, figsize=(5.8, 2.5), sharey=True)
    for ax, (h, lab) in zip(axs, (("b200", "B200"), ("h200", "H200"))):
        rows = [r for r in ix if r["hardware"] == h]
        for j, t in enumerate((25, 50, 100)):
            j = 1.4 * j
            for d, ck, dx in ((1, "network", -0.22), (0, "neutral", 0.22)):
                v = best_at([r for r in rows if r["disagg"] == d], t)
                ax.bar(j + dx, v, width=0.38, color=C[ck])
                ax.text(j + dx, v * 1.12, f"{v:,.0f}", ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color=C[ck])
        ax.set_yscale("log")
        ax.set_ylim(20, 30000)
        ax.set_xticks([0, 1.4, 2.8])
        ax.set_xticklabels(["25", "50", "100"])
        ax.set_xlabel("tok/s per user, at least")
        ax.set_title(lab, fontsize=12, fontweight="bold", loc="left")
        ax.minorticks_off()
    axs[0].set_ylabel("top tok/s per GPU")
    axs[1].text(0.55, 0.93, "split pools", transform=axs[1].transAxes, color=C["network"],
                fontsize=11, fontweight="bold")
    axs[1].text(0.55, 0.80, "colocated", transform=axs[1].transAxes, color=C["neutral"],
                fontsize=11, fontweight="bold")
    sp.save(fig, out / "disagg.svg")


disagg()
for h in ("b200", "h200"):
    rows = [r for r in ix if r["hardware"] == h]
    for t in (25, 50, 100):
        print(f"disagg {h} @{t}: split {best_at([r for r in rows if r['disagg'] == 1], t):.0f} "
              f"colocated {best_at([r for r in rows if r['disagg'] == 0], t):.0f}")

# ================================================ tensor parallel per layer
def tp(k):
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    ax.barh(1, 12, color=C["compute"], height=0.55)
    ax.barh(1, 18, left=12, color=C["compute_l"], height=0.55, ec=C["compute"])
    ax.text(6, 1, "attn 12", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax.text(21, 1, "FFN 18", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(30.5, 1, "30 ms", va="center", fontsize=12, fontweight="bold")
    ax.text(-0.5, 1, "1 GPU", ha="right", va="center", fontsize=12, fontweight="bold")
    if k >= 2:
        ax.barh(0, 1.5, color=C["compute"], height=0.55)
        ax.barh(0, 0.3, left=1.5, color=C["network"], height=0.55)
        ax.barh(0, 2.25, left=1.8, color=C["compute_l"], height=0.55, ec=C["compute"])
        ax.barh(0, 0.3, left=4.05, color=C["network"], height=0.55)
        ax.text(4.8, 0, "4.35 ms = 3.75 compute + 2 × 0.3 AllReduce", va="center", fontsize=11.5,
                fontweight="bold", color=C["network"])
        ax.text(-0.5, 0, "8-way TP", ha="right", va="center", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 36)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Milliseconds per layer, long-prompt prefill (illustrative)")
    sp.save(fig, out / f"tp_step{k}.svg")


for k in (1, 2):
    tp(k)

# ================================== decode-step budget of the reference replica
PARTS = [("GEMM", "compute"), ("attention", "memory"), ("element-wise", "data"),
         ("AllReduce", "network"), ("launch", "neutral")]
COLS = [("Baseline", [34, 20.4, 10.2, 10.2, 10.2]),
        ("INT4 weights", [18.7, 20.4, 10.2, 10.2, 10.2]),
        ("+ fusion, replay", [18.7, 15.3, 3.4, 10.2, 2.0])]


def budget(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    n = min(k, 3)
    for i, (lab, vals) in enumerate(COLS):
        y = 2 - i
        left = 0
        a = 1.0 if i < n else 0.0
        for (pname, ck), v in zip(PARTS, vals):
            ax.barh(y, v, left=left, color=C[ck], height=0.6, alpha=a, ec="white", lw=0.6)

            left += v
        ax.text(-1, y, lab, ha="right", va="center", fontsize=11.5, fontweight="bold", alpha=a)
        ax.text(left + 1, y, f"{sum(vals):.1f} ms", va="center", fontsize=12, fontweight="bold",
                alpha=a, color=C["danger"] if sum(vals) > 50 else C["data"])
    for (pname, ck), xk in zip(PARTS, (0, 14, 28.5, 57, 79)):
        ax.text(xk, 2.62, pname, fontsize=10, color=C[ck], fontweight="bold")
    ax.axvline(50, color=C["danger"], ls="--", lw=1.8)
    ax.text(50.8, -2.3, "50 ms target", color=C["danger"], fontsize=10.5, fontweight="bold")
    if k >= 4:
        ax.barh(-1, 15.2, color=C["data"], height=0.6)
        ax.text(52, -0.85, "15.2 ms per token on average,", va="center", fontsize=11,
                fontweight="bold", color=C["data"])
        ax.text(52, -1.4, "but blocks arrive 53.6 ms apart", va="center", fontsize=11,
                fontweight="bold", color=C["danger"])
        ax.text(-1, -1, "Speculate", ha="right", va="center", fontsize=11.5, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-2.6, 3.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("ms per batch-one decode step, 70B on 8× H100 (constructed)")
    sp.save(fig, out / f"budget_step{k}.svg")


for k in (1, 2, 3, 4):
    budget(k)

# ======================================================= power of two choices
rng = np.random.default_rng(7)


def maxload(n, d, trials):
    res = []
    for _ in range(trials):
        if d == 1:
            res.append(np.bincount(rng.integers(0, n, n), minlength=n).max())
        else:
            load = np.zeros(n, dtype=int)
            a = rng.integers(0, n, n)
            b = rng.integers(0, n, n)
            for i in range(n):
                j = a[i] if load[a[i]] <= load[b[i]] else b[i]
                load[j] += 1
            res.append(load.max())
    return float(np.mean(res))


NS = [10, 100, 1000, 10000, 100000]
R1 = [maxload(n, 1, 20 if n < 1e5 else 5) for n in NS]
R2 = [maxload(n, 2, 20 if n < 1e5 else 5) for n in NS]
print("P2C sim", list(zip(NS, R1, R2)))


def p2c():
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    ax.plot(NS, R1, "o-", color=C["neutral"], lw=2.6, ms=7)
    ax.plot(NS, R2, "o-", color=C["memory"], lw=2.6, ms=7)
    ax.text(NS[-1], R1[-1] + 0.6, f"random: {R1[-1]:.1f}", ha="right", va="bottom", fontsize=11.5,
            fontweight="bold", color=C["neutral"])
    ax.text(NS[-1], R2[-1] - 1.0, f"two probes: {R2[-1]:.1f}", ha="right", va="top", fontsize=11.5,
            fontweight="bold", color=C["memory"])
    ax.set_xscale("log")
    ax.set_xlim(7, 3e5)
    ax.set_xticks([10, 100, 1000, 10000, 100000])
    ax.set_xticklabels(["10", "100", "1k", "10k", "100k"])
    ax.minorticks_off()
    ax.set_ylim(0, max(R1) + 1.5)
    ax.set_xlabel("Replicas n (n equal jobs, no departures)")
    ax.set_ylabel("Busiest replica's jobs")
    sp.save(fig, out / "p2c.svg")


p2c()

# ================================================================== fan-out
def fanout():
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    N = np.logspace(0, 3, 200)
    for q, lab, ck in ((0.01, "leaf slow 1% (p99)", "danger"), (0.001, "0.1% (p99.9)", "compute"),
                       (1e-4, "0.01% (p99.99)", "data")):
        p = 1 - (1 - q) ** N
        ax.plot(N, 100 * p, color=C[ck], lw=2.6)
        v = 100 * (1 - (1 - q) ** 100)
        ax.plot([100], [v], "o", ms=8, color=C[ck], mec="white", zorder=5)
        ax.text(1150, 100 * p[-1], lab, fontsize=10.5, color=C[ck], fontweight="bold", va="center")
        dx, dy, ha = {0.01: (85, 4, "right"), 0.001: (118, 8, "left"), 1e-4: (118, -9, "left")}[q]
        ax.text(dx, v + dy, f"{v:.0f}%" if v >= 5 else f"{v:.1f}%", fontsize=11,
                color=C[ck], fontweight="bold", ha=ha)
    ax.axvline(100, color=C["neutral"], ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.set_ylim(-12, 105)
    ax.set_xlabel("Leaves one request waits for (N)")
    ax.set_ylabel("Requests slow (%)")
    sp.save(fig, out / "fanout.svg")


fanout()

# ============================================================ real traffic
def traffic():
    r = list(csv.DictReader(open(DATA / "azure_llm_conv_2024_10min.csv")))
    t = [datetime.strptime(x["bin10min"], "%Y-%m-%d %H:%M") for x in r]
    v = np.array([int(x["requests"]) for x in r]) / 600.0
    hours = np.array([(x - t[0]).total_seconds() / 3600 for x in t])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.8, 3.0), gridspec_kw={"height_ratios": [1.2, 1]})
    a1.plot(hours / 24, v, color=C["memory"], lw=1.6)
    a1.set_ylabel("req/s", fontsize=11)
    a1.set_xlim(0, 7)
    a1.set_ylim(0, 95)
    a1.set_xticks(np.arange(0, 7) + 0.5)
    a1.set_xticklabels(["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
    a1.text(1.6, 84, f"weekday peak {v.max():.0f} req/s, about 2× the night trough", fontsize=10.5,
            fontweight="bold", color=C["memory"])
    secs = np.array([(x - tc[0]).total_seconds() for x in tc])
    per = np.bincount(secs.astype(int))
    a2.plot(np.arange(len(per)) / 60, per, color=C["compute"], lw=0.8)
    a2.set_xlabel("Minutes (one hour, code trace); top: one week, chat trace, 10-min means")
    a2.set_ylabel("req per s", fontsize=11)
    a2.text(2, per.max() * 0.82, f"mean {per.mean():.1f}/s, busiest second {per.max()}", fontsize=10.5,
            fontweight="bold", color=C["compute"])
    a2.set_xlim(0, 58)
    print(f"traffic: week max {v.max():.1f} min {v.min():.1f}; code per-second mean {per.mean():.2f} "
          f"max {per.max()}")
    sp.save(fig, out / "traffic.svg")


traffic()

# ============================================================ circuit breaker
def breaker():
    fig, ax = plt.subplots(figsize=(5.6, 2.2))
    xs = (0.9, 4.2, 7.5)
    nodes = [(xs[0], "Closed", "data", "normal traffic"),
             (xs[1], "Open", "danger", "fail fast,\nreroute within budget"),
             (xs[2], "Half-open", "compute", "a few probes")]
    for x, lab, ck, sub in nodes:
        ax.add_patch(FancyBboxPatch((x - 0.85, 0.55), 1.7, 0.7, boxstyle="round,pad=0.05",
                                    fc=C[ck + "_l"], ec=C[ck], lw=2))
        ax.text(x, 0.9, lab, ha="center", va="center", fontsize=12.5, fontweight="bold", color=C[ck])
        ax.text(x, 0.3, sub, ha="center", va="top", fontsize=10, color=C["text"])
    for a, b, lab in ((xs[0], xs[1], "errors or\nlatency high"), (xs[1], xs[2], "after an\ninterval")):
        ax.annotate("", xy=(b - 0.95, 0.9), xytext=(a + 0.95, 0.9),
                    arrowprops=dict(arrowstyle="->", lw=1.6))
        ax.text((a + b) / 2, 1.05, lab, ha="center", va="bottom", fontsize=9.5)
    ax.annotate("", xy=(xs[0], 1.35), xytext=(xs[2], 1.35),
                arrowprops=dict(arrowstyle="->", lw=1.4, connectionstyle="arc3,rad=0.28",
                                color=C["data"]))
    ax.text(xs[1], 2.45, "probes succeed", ha="center", fontsize=9.5, color=C["data"],
            fontweight="bold")
    ax.set_xlim(-0.1, 8.5)
    ax.set_ylim(-0.4, 2.7)
    hide(ax)
    sp.save(fig, out / "breaker.svg")


breaker()

# ================================================== synthesis: cost per token
PRICE = 4.00  # $/GPU-hour, Ch. table 25 hypothetical H100 on-demand snapshot
h100 = [r for r in ix if r["hardware"] == "h100"]
SYN = []
for tgt in (13, 50, 100):
    tok = best_at(h100, tgt)
    SYN.append((tgt, tok, PRICE / (tok * 3600) * 1e6))
print("synthesis", SYN)


def synth():
    rows = [(f"{t} tok/s\nper user", c, ck, f"\\${c:,.2f}  ({tok:,.0f} tok/s/GPU)")
            for (t, tok, c), ck in zip(SYN, ("data", "compute", "danger"))]
    sp.hbars(out / "synth_cost.svg", rows, "", log=True, size=(5.8, 1.9),
             xlabel="\\$ per million output tokens, H100 (derived)")


synth()
