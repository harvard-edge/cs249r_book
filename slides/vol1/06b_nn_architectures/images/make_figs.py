#!/usr/bin/env python3
"""Data plots for Lecture 6b (Network Architectures, Part 2).

Every value is sourced from the rendered chapter (Network Architectures),
from MLSysIM, or from a cited outside source. Values marked "derived" are
computed here from a chapter formula and an MLSysIM shape or spec, and the
slide that shows them says so.

  - Context windows (published): BERT 512 positions (Devlin et al. 2019);
    GPT-2 1,024 (Radford et al. 2019); GPT-3 2,048 (Brown et al. 2020);
    GPT-4 Turbo 128K (OpenAI DevDay post, 6 Nov 2023); Claude 2.1 200K
    (Anthropic, 21 Nov 2023); Gemini 1.5 Pro up to 1M in limited preview,
    standard 128K (Google, 15 Feb 2024).
  - RNN state 2 KB for a 512-wide FP32 state (Sec. 6.5.5); GPT-2 XL KV cache
    307.2 KB per token in FP16 (Sec. 6.7.4).
  - GPT-2 XL shape: 48 blocks, width 1,600, 25 heads (MLSysIM GPT2);
    vocabulary 50,257 (Radford et al. 2019; MLSysIM GPT3 vocab_size).
    Block weights 12 d^2 and block MACs 12 S d^2 + 2 S^2 d (Sec. 6.7.3,
    Table 7); crossover at S = 6 d = 9,600 (Sec. 6.7.4).
  - Materialized scores: S^2 per head at 2 bytes, 12 heads (Napkin Math 6.1).
  - Decode intensity 1.0 FLOP/byte in FP16 (Sec. 6.7.4); H100 989 TFLOP/s
    dense FP16 and 3.35 TB/s HBM, PCIe Gen5 x16 64 GB/s (MLSysIM H100).
  - Context memory: Part 1's redraw of the chapter's figure 13 (GPT-2 XL,
    FP16 weights 3 GB, KV 307.2 KB per token, one layer's scores with 25
    heads, one 80 GiB accelerator).
  - KV cache per token, FP16, 2 x layers x key-value width x 2 bytes:
    GPT-2 XL 48 x 1,600; Llama 2 7B 32 x 4,096 (32 key-value heads);
    Llama 3.1 8B 32 x 1,024 (8 key-value heads of 128); Llama 2 70B
    80 x 1,024 (8 of 128; 80 x 8,192 without sharing); PaLM 540B 118 x 256
    (one key-value head of 256). MLSysIM Llama2_7B, Llama3_8B, Llama2_70B,
    PaLM_540B; Ainslie et al. 2023; Shazeer 2019; Chowdhery et al. 2022.
  - Recommender request (illustrative): 26 categorical features as in
    Criteo (Naumov et al. 2019), one 128-wide FP32 row each, 512 B per row
    (Sec. 6.8.3).
  - Capacity wall: Napkin Math 6.2.
  - One 1M-token request on a model of Llama 3.1 8B's shape (illustrative;
    its card states a 128K window): BF16 weights 2 x 8.03B; KV cache
    131,072 B per token; scores 2 B x S^2 per head per layer.
  - Fiction.LiveBench, DeepSeek-R1-0528 (MLSysIM literature series
    fiction_livebench_by_length; Epoch AI, CC BY 4.0).

Run from this folder: python3 make_figs.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

out = Path(__file__).parent
C = sp.C
MLSYSIM = Path(__file__).resolve().parents[4] / "mlsysim"
plt = sp.plt


def save(fig, path, margins):
    """Save with fixed margins so every step of a build keeps one canvas."""
    fig.subplots_adjust(**margins)
    fig.savefig(path, format="svg", facecolor="white")
    plt.close(fig)


# --- Shapes and specs -------------------------------------------------------
D_XL, L_XL, VOCAB = 1600, 48, 50257          # GPT-2 XL (MLSysIM GPT2)
H100_FLOPS, H100_BW, PCIE_BW = 989e12, 3.35e12, 64e9   # MLSysIM H100
KV_XL = 2 * L_XL * D_XL * 2                  # 307,200 B per token


# --- Hook: published context windows ---------------------------------------
def context_windows():
    fig, ax = plt.subplots(figsize=(5.8, 1.95))
    pts = [  # (year, tokens, label, label offset in points, ha, va)
        (2018.8, 512, "BERT 512", (0, -9), "center", "top"),
        (2019.15, 1024, "GPT-2 1,024", (0, 9), "center", "bottom"),
        (2020.4, 2048, "GPT-3 2,048", (9, 0), "left", "center"),
        (2023.85, 128e3, "GPT-4 Turbo 128K", (-9, -5), "right", "center"),
        (2023.89, 200e3, "Claude 2.1 200K", (-9, 7), "right", "center"),
        (2024.12, 1e6, "Gemini 1.5 Pro 1M (preview)", (-9, 2), "right", "center"),
    ]
    for x, y, label, off, ha, va in pts:
        col = C["danger"] if y >= 1e5 else C["memory"]
        ax.plot([x], [y], "o", ms=10, color=col, mec="white", mew=1.5, zorder=5)
        ax.annotate(label, (x, y), xytext=off, textcoords="offset points",
                    ha=ha, va=va, fontsize=10.5, fontweight="bold", color=col)
    ax.text(2021.75, 8e3, "GPT-3 to Gemini 1.5:\nabout 500× in four years",
            color=C["text"], fontsize=11.5, fontweight="bold", ha="center")
    ax.set_yscale("log")
    ax.set_xlim(2018.3, 2024.5)
    ax.set_ylim(100, 4e6)
    ax.set_xticks([2018, 2019, 2020, 2021, 2022, 2023, 2024])
    ax.set_xticklabels(["2018", "2019", "2020", "2021", "2022", "2023", "2024"])
    ax.set_yticks([1e3, 1e4, 1e5, 1e6])
    ax.set_yticklabels(["1K", "10K", "100K", "1M"])
    ax.minorticks_off()
    ax.set_ylabel("Context window\n(tokens, log scale)")
    ax.grid(color=C["grid"], lw=0.6)
    save(fig, out / "context_windows.svg",
         dict(left=0.15, right=0.98, bottom=0.14, top=0.97))


# --- Recall: fixed RNN state against a growing KV cache (Part 1's plot) -----
def state_memory():
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    s = np.logspace(1, 4, 50)
    ax.plot(s, np.full_like(s, 2e3), color=C["memory"], lw=3.5)
    ax.text(12, 2e3 * 1.9, "RNN state: 2 KB at any length", color=C["memory"],
            fontsize=12, fontweight="bold")
    ax.plot(s, KV_XL * s, color=C["danger"], lw=3.5)
    ax.text(10500, 3.07e9, "3.1 GB", color=C["danger"], fontsize=12,
            fontweight="bold", va="center")
    ax.text(1500, 1e5, "Transformer: 307.2 KB\nkept per token (GPT-2 XL)",
            color=C["danger"], fontsize=12, fontweight="bold", ha="center",
            va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 1e4)
    ax.set_ylim(5e2, 3e10)
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10", "100", "1,000", "10,000"])
    ax.set_yticks([1e3, 1e6, 1e9])
    ax.set_yticklabels(["1 KB", "1 MB", "1 GB"])
    ax.minorticks_off()
    ax.set_xlabel("Sequence length (tokens, log scale)")
    ax.set_ylabel("State per sequence")
    ax.grid(color=C["grid"], lw=0.6)
    save(fig, out / "state_memory.svg",
         dict(left=0.14, right=0.86, bottom=0.21, top=0.96))


# --- Segment 1: GPT-2 XL's weights from the block rule (derived) -----------
def gpt2xl_params():
    attn = 4 * D_XL ** 2 * L_XL / 1e9          # 0.49B
    mlp = 8 * D_XL ** 2 * L_XL / 1e9           # 0.98B
    table = VOCAB * D_XL / 1e9                 # 0.08B
    fig, ax = plt.subplots(figsize=(5.8, 1.55))
    segs = [(attn, "compute", f"Attention\nprojections {attn:.2f}B"),
            (mlp, "memory", f"MLP {mlp:.2f}B"),
            (table, "network", "")]
    left = 0
    for w, ck, txt in segs:
        ax.barh(0, w, left=left, color=C[ck], height=0.55, edgecolor="white", lw=1.5)
        if txt:
            ax.text(left + w / 2, 0, txt, ha="center", va="center", color="white",
                    fontsize=11.5, fontweight="bold", linespacing=1.0)
        left += w
    ax.annotate(f"Token table {table:.2f}B", (left - table / 2, -0.28),
                xytext=(left - 0.02, -0.72), ha="right", va="center",
                fontsize=11, fontweight="bold", color=C["network"],
                arrowprops=dict(arrowstyle="-", color=C["network"], lw=1.2))
    ax.text(left + 0.03, 0, f"= {left:.2f}B", va="center", fontsize=13,
            fontweight="bold", color=C["text"])
    ax.plot([1.5, 1.5], [-0.42, 0.42], color=C["danger"], lw=2.5)
    ax.text(1.5, 0.5, "published 1.5B", ha="center", va="bottom", fontsize=11,
            fontweight="bold", color=C["danger"])
    ax.set_xlim(0, 1.9)
    ax.set_ylim(-0.95, 0.85)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 0.5, 1.0, 1.5])
    ax.set_xticklabels(["0", "0.5B", "1.0B", "1.5B"])
    ax.set_xlabel("Weights (48 blocks × 12 d², d = 1,600, plus the token table)")
    save(fig, out / "gpt2xl_params.svg",
         dict(left=0.03, right=0.97, bottom=0.30, top=0.97))


# --- Segment 2: where a block's MACs go as context grows (derived) ---------
def block_macs(step):
    fig, ax = plt.subplots(figsize=(5.8, 2.45))
    s = np.logspace(np.log10(128), np.log10(2e5), 200)
    ax.plot(s, 12 * s * D_XL ** 2, color=C["compute"], lw=3.2)
    ax.plot(s, 2 * s ** 2 * D_XL, color=C["memory"], lw=3.2)
    ax.text(140, 3e11, "Projections + MLP: 12 S d²", color=C["compute"],
            fontsize=11.5, fontweight="bold")
    ax.text(1.4e4, 3e10, "Attention\nproducts: 2 S² d", color=C["memory"],
            fontsize=11.5, fontweight="bold", va="top", linespacing=1.05)
    notes = [  # shown from step k: (x, label, color, text position, ha, va)
        (2, 1024, "1,024 tokens:\nattention is about 10%", C["text"],
         (1150, 1.2e9), "left", "top"),
        (3, 6 * D_XL, "6d = 9,600:\nequal", C["danger"],
         (7500, 1.5e12), "right", "bottom"),
        (4, 1e5, "100K tokens: attention\nis 10× the rest", C["memory"],
         (5.5e4, 1.8e14), "center", "bottom"),
    ]
    for k, x, label, col, pos, ha, va in notes:
        alpha = 1.0 if step >= k else 0.0
        lo, hi = sorted((2 * x * x * D_XL, 12 * x * D_XL ** 2))
        if k == 3:
            ax.plot([x], [hi], "o", ms=10, color=col, mec="white", zorder=5,
                    alpha=alpha)
        else:
            ax.plot([x, x], [lo, hi], color=col, lw=1.4, ls="--", alpha=alpha)
            for y in (lo, hi):
                ax.plot([x], [y], "o", ms=7, color=col, mec="white", zorder=5,
                        alpha=alpha)
        ax.text(*pos, label, color=col, fontsize=11, fontweight="bold",
                ha=ha, va=va, alpha=alpha, linespacing=1.05)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(128, 2e5)
    ax.set_ylim(1e7, 1e17)
    ax.set_xticks([128, 1024, 9600, 1e5])
    ax.set_xticklabels(["128", "1,024", "9,600", "100K"])
    ax.set_yticks([1e9, 1e12, 1e15])
    ax.set_yticklabels(["10⁹", "10¹²", "10¹⁵"])
    ax.minorticks_off()
    ax.set_xlabel("Context S (tokens, log scale)")
    ax.set_ylabel("MACs per block\n(log scale)")
    ax.grid(color=C["grid"], lw=0.6)
    save(fig, out / f"block_macs_step{step}.svg",
         dict(left=0.13, right=0.97, bottom=0.20, top=0.97))


# --- Segment 2: the S-squared parabola (Part 1's plot, column size) --------
def quadratic():
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    s = np.linspace(0, 8800, 200)
    ax.plot(s, s ** 2 * 24 / 1e9, color=C["compute"], lw=3.5)
    for x, label, col in [(4096, "4K: 0.40 GB", C["data"]),
                          (8192, "8K: 1.61 GB\n= 4×", C["danger"])]:
        y = x ** 2 * 24 / 1e9
        ax.plot([x, x], [0, y], ls="--", color=col, lw=1.2)
        ax.plot([x], [y], "o", ms=10, color=col, mec="white", mew=1.5, zorder=5)
        ax.text(x - 400, y + 0.05, label, ha="right", va="bottom", color=col,
                fontsize=12, fontweight="bold")
    ax.set_xlim(0, 8800)
    ax.set_ylim(0, 2.1)
    ax.set_xticks([0, 4096, 8192])
    ax.set_xticklabels(["0", "4K", "8K"])
    ax.set_xlabel("Context (tokens)")
    ax.set_ylabel("Scores, one layer (GB)")
    ax.text(0.03, 0.97, "12 heads, FP16", transform=ax.transAxes,
            fontsize=10, color=C["neutral"], va="top")
    ax.grid(color=C["grid"], lw=0.6)
    save(fig, out / "quadratic.svg",
         dict(left=0.22, right=0.96, bottom=0.19, top=0.97))


# --- Segment 3: decode intensity against the H100 ridge --------------------
def decode_ridge():
    ridge = H100_FLOPS / H100_BW
    fig, ax = plt.subplots(figsize=(5.8, 1.45))
    rows = [("Decode, batch one, FP16", 1.0, "memory", "1 FLOP/byte"),
            ("H100 balance (ridge)", ridge, "compute", f"{ridge:.0f} FLOP/byte")]
    for i, (label, v, ck, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v - 0.3, left=0.3, color=C[ck], height=0.6)
        ax.text(0.27, y, label, ha="right", va="center", fontsize=12.5,
                fontweight="bold", color=C["text"])
        inside = v > 50
        ax.text(v / 1.15 if inside else v * 1.2, y, txt, va="center",
                ha="right" if inside else "left", fontsize=12.5,
                fontweight="bold", color="white" if inside else C[ck])
    ax.set_xscale("log")
    ax.set_xlim(0.3, 4000)
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.set_xlabel("FLOP per byte (log scale)")
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    save(fig, out / "decode_ridge.svg",
         dict(left=0.40, right=0.97, bottom=0.33, top=0.97))


# --- Segment 3: three memories as context grows (Part 1's redraw) ----------
def context_memory(step):
    fig, ax = plt.subplots(figsize=(5.8, 2.35))
    s = np.logspace(np.log10(512), np.log10(131072), 200)
    dev = 80 * 2 ** 30 / 1e9
    ax.axhline(dev, color=C["danger"], lw=1.3, ls=":")
    ax.text(600, dev / 1.5, "One 80 GB accelerator", color=C["danger"], fontsize=11,
            va="top")
    ax.plot(s, np.full_like(s, 3.0), color=C["memory"], lw=3.2)
    ax.text(600, 3.0 * 1.4, "Weights, 3 GB (fixed)", color=C["memory"],
            fontsize=12, fontweight="bold")
    if step >= 2:
        ax.plot(s, KV_XL * s / 1e9, color=C["compute"], lw=3.2)
        x = 3.0e9 / KV_XL
        ax.plot([x], [3.0], "o", ms=9, color=C["compute"], mec="white", zorder=5)
        ax.text(x * 1.25, 0.06, "KV cache passes\nthe weights near 10K",
                color=C["compute"], fontsize=11, fontweight="bold")
    if step >= 3:
        ax.plot(s, 25 * s ** 2 * 2 / 1e9, color=C["danger"], lw=3.2, ls="--")
        x = np.sqrt(80 * 2 ** 30 / 50)
        ax.plot([x], [dev], "o", ms=9, color=C["danger"], mec="white", zorder=5)
        ax.text(x / 1.9, dev * 2.2, "One layer's scores\nfill it near 41K",
                color=C["danger"], fontsize=11, fontweight="bold", ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(512, 131072)
    ax.set_ylim(0.01, 2000)
    ax.set_xticks([512, 2048, 8192, 32768, 131072])
    ax.set_xticklabels(["512", "2K", "8K", "32K", "128K"])
    ax.set_yticks([0.01, 1, 100])
    ax.set_yticklabels(["10 MB", "1 GB", "100 GB"])
    ax.minorticks_off()
    ax.set_xlabel("Context length (tokens, log scale)")
    ax.set_ylabel("Memory (log scale)")
    ax.grid(color=C["grid"], lw=0.6)
    save(fig, out / f"context_memory_step{step}.svg",
         dict(left=0.14, right=0.86, bottom=0.21, top=0.96))


# --- Segment 3: KV cache per token across decoders (derived) ---------------
def kv_per_token():
    def kv(layers, width):
        return 2 * layers * width * 2 / 1e3          # KB, FP16

    rows = [  # label, KB, color, value text, ghost KB or None
        ("GPT-2 XL (1.5B)", kv(48, 1600), "neutral", "307 KB: 25 own heads", None),
        ("Llama 2 7B", kv(32, 4096), "danger", "524 KB: 32 own heads", None),
        ("Llama 3.1 8B", kv(32, 1024), "memory", "131 KB: 8 shared", None),
        ("Llama 2 70B", kv(80, 1024), "memory", "328 KB: 8 shared", kv(80, 8192)),
        ("PaLM 540B", kv(118, 256), "memory", "121 KB: 1 shared", None),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 2.45))
    n = len(rows)
    for i, (label, v, ck, txt, ghost) in enumerate(rows):
        y = n - 1 - i
        if ghost:
            ax.barh(y, ghost - 30, left=30, color="white", edgecolor=C["danger"],
                    ls="--", lw=1.4, height=0.62)
            ax.text(ghost * 1.1, y, "2.6 MB\nunshared", va="center", fontsize=10.5,
                    color=C["danger"], fontweight="bold", linespacing=1.0)
        ax.barh(y, v - 30, left=30, color=C[ck], height=0.62)
        ax.text(27, y, label, ha="right", va="center", fontsize=12,
                fontweight="bold", color=C["text"])
        if ghost:
            ax.text(36, y, txt, va="center", fontsize=11, fontweight="bold",
                    color="white")
        else:
            ax.text(v * 1.1, y, txt, va="center", fontsize=11, fontweight="bold",
                    color=C[ck])
    ax.set_xscale("log")
    ax.set_xlim(30, 9000)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([100, 1000])
    ax.set_xticklabels(["100 KB", "1 MB"])
    ax.minorticks_off()
    ax.set_xlabel("KV cache per token, FP16 (log scale)")
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    save(fig, out / "kv_per_token.svg",
         dict(left=0.27, right=0.98, bottom=0.20, top=0.98))


# --- Segment 4: request rate each memory path could feed (derived) ---------
def recsys_rate():
    req_bytes = 26 * 128 * 4                      # 13,312 B per request
    hbm, pcie = H100_BW / req_bytes, PCIE_BW / req_bytes
    fig, ax = plt.subplots(figsize=(2.6, 2.55))
    for x, v, ck, txt in [(0, hbm, "memory", f"{hbm / 1e6:.0f}M"),
                          (1, pcie, "network", f"{pcie / 1e6:.1f}M")]:
        ax.bar(x, v, color=C[ck], width=0.62)
        ax.text(x, v * 1.4, txt, ha="center", va="bottom", fontsize=12.5,
                fontweight="bold", color=C[ck])
    ax.text(1, 3.2e7, f"{hbm / pcie:.0f}× fewer", ha="center", fontsize=12.5,
            fontweight="bold", color=C["danger"])
    ax.set_yscale("log")
    ax.set_ylim(1e6, 3e9)
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["HBM\n3.35 TB/s", "PCIe Gen5\n64 GB/s"])
    ax.set_yticks([1e6, 1e7, 1e8, 1e9])
    ax.set_yticklabels(["1M", "10M", "100M", "1B"])
    ax.minorticks_off()
    ax.set_ylabel("Requests/s, upper bound")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    save(fig, out / "recsys_rate.svg",
         dict(left=0.30, right=0.97, bottom=0.22, top=0.97))


# --- Segment 4: the capacity wall (Part 1's plot, column size) -------------
def capacity():
    fig, ax = plt.subplots(figsize=(3.6, 1.85))
    rows = [("One table", 51.2, "memory", "51.2 GB"),
            ("Two tables", 102.4, "danger", "102.4 GB")]
    for i, (label, v, ck, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=C[ck], height=0.6)
        ax.text(-2, y, label, ha="right", va="center", fontsize=12,
                fontweight="bold", color=C["text"])
        if v < 80:
            ax.text(v - 2, y, txt, va="center", ha="right", fontsize=12,
                    fontweight="bold", color="white")
        else:
            ax.text(v + 3, y, txt, va="center", fontsize=12, fontweight="bold",
                    color=C[ck])
    ax.axvline(80, color=C["danger"], lw=2, ls="--")
    ax.text(78, 1.62, "80 GB A100", ha="right", fontsize=11, color=C["danger"],
            fontweight="bold")
    ax.set_xlim(0, 150)
    ax.set_ylim(-0.5, 1.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xlabel("GB (100M rows, 128-wide, FP32)")
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    save(fig, out / "capacity.svg",
         dict(left=0.27, right=0.97, bottom=0.27, top=0.97))


# --- Synthesis: one 1M-token request (derived, illustrative shape) ---------
def million_request(step):
    s = 1e6
    weights = 2 * 8.03e9 / 1e9                     # 16.1 GB, BF16
    kv = s * 2 * 32 * 1024 * 2 / 1e9               # 131 GB
    scores = 2 * s ** 2 / 1e9                      # 2,000 GB per head per layer
    rows = [("Weights, BF16", weights, "memory", f"{weights:.1f} GB"),
            ("KV cache, 1M tokens", kv, "compute", f"{kv:.0f} GB"),
            ("Scores if stored", scores, "danger", "2 TB per head, per layer")]
    fig, ax = plt.subplots(figsize=(5.8, 1.95))
    n = len(rows)
    for i, (label, v, ck, txt) in enumerate(rows):
        y = n - 1 - i
        shown = i < step
        ghost = i == 2
        if shown:
            if ghost:
                ax.barh(y, v - 1, left=1, color=C["danger_l"], edgecolor=C[ck],
                        ls="--", lw=1.4, height=0.6)
            else:
                ax.barh(y, v - 1, left=1, color=C[ck], height=0.6)
            if ghost:
                ax.text(1.12, y, txt, va="center", ha="left", fontsize=11,
                        fontweight="bold", color=C["danger"], zorder=6,
                        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
            elif v < 80:
                ax.text(v / 1.12, y, txt, va="center", ha="right", fontsize=12,
                        fontweight="bold", color="white")
            else:
                ax.text(v * 1.15, y, txt, va="center", fontsize=12,
                        fontweight="bold", color=C[ck])
        ax.text(0.92, y, label, ha="right", va="center", fontsize=12,
                fontweight="bold", color=C["text"] if shown else "white")
    ax.axvline(80, color=C["danger"], lw=2, ls=":")
    ax.text(88, n - 0.42, "80 GB device", ha="left", va="bottom", fontsize=10.5,
            color=C["danger"], fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(1, 8000)
    ax.set_ylim(-0.55, n - 0.05)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1 GB", "10 GB", "100 GB", "1 TB"])
    ax.minorticks_off()
    ax.set_xlabel("Memory for one request (log scale)")
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    save(fig, out / f"million_request_step{step}.svg",
         dict(left=0.30, right=0.97, bottom=0.25, top=0.88))


# --- Open questions: comprehension against context (published) -------------
def fiction_livebench():
    path = MLSYSIM / "mlsysim" / "literature" / "data" / "series" / \
        "fiction_livebench_by_length.csv"
    cols = [("score_400", 400), ("score_1k", 1e3), ("score_2k", 2e3),
            ("score_4k", 4e3), ("score_8k", 8e3), ("score_16k", 16e3),
            ("score_32k", 32e3), ("score_60k", 60e3)]
    with open(path, newline="") as fh:
        row = next(r for r in csv.DictReader(fh)
                   if r["model"] == "chutes/DeepSeek-R1-0528")
    xs = [x for _, x in cols]
    ys = [float(row[k]) for k, _ in cols]
    fig, ax = plt.subplots(figsize=(2.9, 2.2))
    ax.plot(xs, ys, color=C["memory"], lw=3, marker="o", ms=6)
    ax.text(xs[0], ys[0] + 0.04, f"{ys[0]:.3f}", ha="left", va="bottom",
            fontsize=11, fontweight="bold", color=C["memory"])
    ax.text(xs[-1], ys[-1] - 0.05, f"{ys[-1]:.3f}", ha="right", va="top",
            fontsize=11, fontweight="bold", color=C["danger"])
    ax.set_xscale("log")
    ax.set_xlim(300, 8e4)
    ax.set_ylim(0.4, 1.05)
    ax.set_xticks([1e3, 1e4])
    ax.set_xticklabels(["1K", "10K"])
    ax.minorticks_off()
    ax.set_xlabel("Tokens of story in context")
    ax.set_ylabel("Score")
    ax.grid(color=C["grid"], lw=0.6)
    save(fig, out / "fiction_livebench.svg",
         dict(left=0.22, right=0.95, bottom=0.22, top=0.96))


context_windows()
state_memory()
gpt2xl_params()
for k in (1, 2, 3, 4):
    block_macs(k)
quadratic()
decode_ridge()
for k in (1, 2, 3):
    context_memory(k)
kv_per_token()
recsys_rate()
capacity()
for k in (1, 2, 3):
    million_request(k)
fiction_livebench()


# --- The toy attention head: values printed for the slide and notes --------
q = np.array([[1, 0], [0, 1], [1, 1], [1, 0]], float)
k = np.array([[2, 0], [0, 2], [1, 1], [0, 0]], float)
v = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], float)
scores = q @ k.T / np.sqrt(2)
w = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
print("toy scores\n", scores.round(2))
print("toy weights\n", w.round(2))
print("toy outputs\n", (w @ v).round(2))
print("figures written")
