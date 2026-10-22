#!/usr/bin/env python3
"""Data plots for Lecture 6b (Network Architectures, Part 2).

Every plotted value and every number in a label comes from calcs.py, which
reads MLSysIM: the model registry (GPT-2 XL, Llama 2 and 3.1, PaLM, OPT-13B),
the H100 and A100, the attention, KV-cache, decode, and recommender scenarios,
and the cited series in Literature (published context windows,
FlashAttention's Tables 20 and 21, Fiction.LiveBench). Positions, axis
limits, and label offsets are layout and stay here.

Run from this folder: python3 make_figs.py (values: calcs.py)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcs as cc  # noqa: E402  (records every value)
from mlsysim.core.series_loader import load_series  # noqa: E402
from deckvals import val, num  # noqa: E402

out = Path(__file__).parent
C = sp.C
plt = sp.plt


def save(fig, path, margins):
    """Save with fixed margins so every step of a build keeps one canvas."""
    fig.subplots_adjust(**margins)
    fig.savefig(path, format="svg", facecolor="white")
    plt.close(fig)


# --- Shapes and specs (calcs.py, from MLSysIM) ------------------------------
KV_XL = cc.m(cc.KV_XL.per_token, "byte")     # bytes per kept token, FP16
KI = cc.m(cc.units.Q_(1, "KiB"), "byte")     # tokens per "K" in a power-of-two length


def k_label(tokens):
    """1,000,000 -> 1M, 128,000 -> 128K: the short form the plots print."""
    return f"{tokens / 1e6:.0f}M" if tokens >= 1e6 else f"{tokens / 1e3:.0f}K"


def ki_label(tokens):
    """4,096 -> 4K: a power-of-two length in binary thousands."""
    return f"{tokens / KI:.0f}K"


# --- Hook: published context windows ---------------------------------------
def context_windows():
    fig, ax = plt.subplots(figsize=(5.8, 1.95))
    gpt4t, claude = num("gpt4t_window"), num("claude21_window")
    gemini = cc.WINDOWS.Gemini15ProPreviewTokens
    pts = [  # (year as a layout position, tokens, label, label offset in points, ha, va)
        (2018.8, num("bert_window"), f"BERT {val('bert_window')}", (0, -9), "center", "top"),
        (2019.15, num("gpt2_window"), f"GPT-2 {val('gpt2_window')}", (0, 9), "center", "bottom"),
        (2020.4, num("gpt3_window"), f"GPT-3 {val('gpt3_window')}", (9, 0), "left", "center"),
        (2023.85, gpt4t, f"GPT-4 Turbo {k_label(gpt4t)}", (-9, -5), "right", "center"),
        (2023.89, claude, f"Claude 2.1 {k_label(claude)}", (-9, 7), "right", "center"),
        (2024.12, gemini, f"Gemini 1.5 Pro {k_label(gemini)} (preview)", (-9, 2), "right", "center"),
    ]
    for x, y, label, off, ha, va in pts:
        col = C["danger"] if y > num("gpt3_window") else C["memory"]
        ax.plot([x], [y], "o", ms=10, color=col, mec="white", mew=1.5, zorder=5)
        ax.annotate(label, (x, y), xytext=off, textcoords="offset points",
                    ha=ha, va=va, fontsize=10.5, fontweight="bold", color=col)
    ax.text(2021.75, 8e3, f"GPT-3 to Gemini 1.5:\nabout {val('window_growth_round')}× in four years",
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
    rnn = cc.m(cc.Scenarios.RecurrentState512Fp32.state_bytes, "byte")
    ax.plot(s, np.full_like(s, rnn), color=C["memory"], lw=3.5)
    ax.text(12, rnn * 1.9, f"RNN state: {val('rnn_state_kb')} KB at any length", color=C["memory"],
            fontsize=12, fontweight="bold")
    ax.plot(s, KV_XL * s, color=C["danger"], lw=3.5)
    ax.text(10500, cc.m(cc.KV_XL_10K.total, "byte"), f"{val('kv_10k_gb')} GB", color=C["danger"], fontsize=12,
            fontweight="bold", va="center")
    ax.text(1500, 1e5, f"Transformer: {val('kv_xl_kb')} KB\nkept per token (GPT-2 XL)",
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
    attn = cc.BLOCK.attention_weights / 1e9
    mlp = cc.BLOCK.mlp_weights / 1e9
    table = cc.BLOCK.token_table / 1e9
    published = num("xl_params_b")
    fig, ax = plt.subplots(figsize=(5.8, 1.3))
    segs = [(attn, "compute", f"Attention\nprojections {val('attn_proj_b')}B"),
            (mlp, "memory", f"MLP {val('mlp_b')}B"),
            (table, "network", "")]
    left = 0
    for w, ck, txt in segs:
        ax.barh(0, w, left=left, color=C[ck], height=0.55, edgecolor="white", lw=1.5)
        if txt:
            ax.text(left + w / 2, 0, txt, ha="center", va="center", color="white",
                    fontsize=11.5, fontweight="bold", linespacing=1.0)
        left += w
    ax.annotate(f"Token table {val('table_b')}B", (left - table / 2, -0.28),
                xytext=(left - 0.02, -0.72), ha="right", va="center",
                fontsize=11, fontweight="bold", color=C["network"],
                arrowprops=dict(arrowstyle="-", color=C["network"], lw=1.2))
    ax.text(left + 0.03, 0, f"= {val('rule_b')}B", va="center", fontsize=13,
            fontweight="bold", color=C["text"])
    ax.plot([published, published], [-0.42, 0.42], color=C["danger"], lw=2.5)
    ax.text(published, 0.5, f"published {val('xl_params_b')}B", ha="center", va="bottom", fontsize=11,
            fontweight="bold", color=C["danger"])
    ax.set_xlim(0, 1.9)
    ax.set_ylim(-0.95, 0.85)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 0.5, 1.0, 1.5])
    ax.set_xticklabels(["0", "0.5B", "1.0B", "1.5B"])
    save(fig, out / "gpt2xl_params.svg",
         dict(left=0.03, right=0.97, bottom=0.17, top=0.97))


# --- Segment 2: where a block's MACs go as context grows (derived) ---------
def block_macs(step):
    fig, ax = plt.subplots(figsize=(5.8, 2.45))
    s = np.logspace(np.log10(128), np.log10(2e5), 200)
    blocks = [cc.BLOCK.variant(context_tokens=x) for x in s]
    ax.plot(s, [b.dense_macs for b in blocks], color=C["compute"], lw=3.2)
    ax.plot(s, [b.attention_macs for b in blocks], color=C["memory"], lw=3.2)
    ax.text(140, 3e11, "Projections + MLP: 12 S d²", color=C["compute"],
            fontsize=11.5, fontweight="bold")
    ax.text(1.4e4, 3e10, "Attention\nproducts: 2 S² d", color=C["memory"],
            fontsize=11.5, fontweight="bold", va="top", linespacing=1.05)
    notes = [  # shown from step k: (block, label, color, text position, ha, va)
        (2, cc.BLOCK, f"{val('gpt2_window')} tokens:\nattention is about {val('attn_share_pct')}%", C["text"],
         (1150, 1.2e9), "left", "top"),
        (3, cc.BLOCK.variant(context_tokens=round(cc.BLOCK.crossover)), f"6d = {val('crossover')}:\nequal", C["danger"],
         (7500, 1.5e12), "right", "bottom"),
        (4, cc.BLOCK_LONG, f"{val('long_tokens_k')}K tokens: attention\nis {val('long_ratio')}× the rest", C["memory"],
         (5.5e4, 1.8e14), "center", "bottom"),
    ]
    for k, b, label, col, pos, ha, va in notes:
        alpha = 1.0 if step >= k else 0.0
        x = b.context_tokens
        lo, hi = sorted((b.attention_macs, b.dense_macs))
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
    ax.set_xticks([128, num("gpt2_window"), num("crossover"), num("long_tokens")])
    ax.set_xticklabels(["128", val("gpt2_window"), val("crossover"), f"{val('long_tokens_k')}K"])
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
    per_pair = cc.m(cc.SCORES.heads * cc.SCORES.bytes_per_score, "GB")
    ax.plot(s, s ** 2 * per_pair, color=C["compute"], lw=3.5)
    four, eight = cc.SCORES_4K, cc.SCORES_8K
    growth = cc.m(eight.layer_bytes / four.layer_bytes, "")
    for sc, label, col in [(four, f"{ki_label(four.seq_len)}: {val('sc4k_gb')} GB", C["data"]),
                           (eight, f"{ki_label(eight.seq_len)}: {val('sc8k_gb')} GB\n= {growth:.0f}×", C["danger"])]:
        x, y = sc.seq_len, cc.m(sc.layer_bytes, "GB")
        ax.plot([x, x], [0, y], ls="--", color=col, lw=1.2)
        ax.plot([x], [y], "o", ms=10, color=col, mec="white", mew=1.5, zorder=5)
        ax.text(x - 400, y + 0.05, label, ha="right", va="bottom", color=col,
                fontsize=12, fontweight="bold")
    ax.set_xlim(0, 8800)
    ax.set_ylim(0, 2.1)
    ax.set_xticks([0, four.seq_len, eight.seq_len])
    ax.set_xticklabels(["0", ki_label(four.seq_len), ki_label(eight.seq_len)])
    ax.set_xlabel("Context (tokens)")
    ax.set_ylabel("Scores, one layer (GB)")
    ax.text(0.03, 0.97, f"{val('sc_heads')} heads, FP16", transform=ax.transAxes,
            fontsize=10, color=C["neutral"], va="top")
    ax.grid(color=C["grid"], lw=0.6)
    save(fig, out / "quadratic.svg",
         dict(left=0.22, right=0.96, bottom=0.19, top=0.97))


# --- Segment 3: decode intensity against the H100 ridge --------------------
def decode_ridge():
    ridge = cc.m(cc.ROOF.ridge, "flop/byte")
    fig, ax = plt.subplots(figsize=(3.4, 1.1))
    rows = [("Decode,\nbatch 1", cc.m(cc.ROOF.point_intensity, "flop/byte"), "memory", f"{val('decode_i')} FLOP/byte"),
            ("H100\nridge", ridge, "compute", f"{val('ridge')} FLOP/byte")]
    for i, (label, v, ck, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v - 0.3, left=0.3, color=C[ck], height=0.6)
        ax.text(0.26, y, label, ha="right", va="center", fontsize=11.5,
                fontweight="bold", color=C["text"], linespacing=1.0)
        inside = v > 50
        ax.text(v / 1.15 if inside else v * 1.2, y, txt, va="center",
                ha="right" if inside else "left", fontsize=11.5,
                fontweight="bold", color="white" if inside else C[ck])
    ax.set_xscale("log")
    ax.set_xlim(0.3, 1000)
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
         dict(left=0.26, right=0.92, bottom=0.40, top=0.98))


# --- Segment 3: three memories as context grows (Part 1's redraw) ----------
def context_memory(step):
    fig, ax = plt.subplots(figsize=(5.8, 2.35))
    s = np.logspace(np.log10(512), np.log10(131072), 200)
    ctx = cc.CTX
    dev = cc.m(ctx.hardware.memory.capacity, "GB")   # the device's bytes, in decimal GB
    weights = cc.m(ctx.weights, "GB")
    ax.axhline(dev, color=C["danger"], lw=1.3, ls=":")
    ax.text(600, dev / 1.5, f"One {val('dev_gb')} GB accelerator", color=C["danger"], fontsize=11,
            va="top")
    ax.plot(s, np.full_like(s, weights), color=C["memory"], lw=3.2)
    ax.text(600, weights * 1.4, f"Weights, {val('ctx_weights_gb')} GB (fixed)", color=C["memory"],
            fontsize=12, fontweight="bold")
    if step >= 2:
        ax.plot(s, cc.m(ctx.kv_per_token, "GB") * s, color=C["compute"], lw=3.2)
        x = ctx.kv_crossing
        ax.plot([x], [weights], "o", ms=9, color=C["compute"], mec="white", zorder=5)
        ax.text(x * 1.25, 0.06, f"KV cache passes\nthe weights near {val('ctx_kv_cross_k')}K",
                color=C["compute"], fontsize=11, fontweight="bold")
    if step >= 3:
        ax.plot(s, cc.m(ctx.score_bytes_per_pair, "GB") * s ** 2, color=C["danger"], lw=3.2, ls="--")
        x = ctx.score_crossing
        ax.plot([x], [dev], "o", ms=9, color=C["danger"], mec="white", zorder=5)
        ax.text(x / 1.9, dev * 2.2, f"One layer's scores\nfill it near {val('ctx_score_cross_k')}K",
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
    unshared = cc.L70.model_copy(update={"kv_heads": cc.L70.heads})
    rows = [  # label, KB, color, value text, ghost KB or None
        (f"GPT-2 XL ({val('xl_params_b')}B)", cc.kv_kb(cc.XL), "neutral",
         f"{val('kv_xl_kb_round')} KB: {val('xl_heads')} own heads", None),
        ("Llama 2 7B", cc.kv_kb(cc.L7), "danger", f"{val('kv_l7_kb')} KB: {val('l7_heads')} own heads", None),
        ("Llama 3.1 8B", cc.kv_kb(cc.L8), "memory", f"{val('kv_l8_kb')} KB: {val('l8_kv_heads')} shared", None),
        ("Llama 2 70B", cc.kv_kb(cc.L70), "memory", f"{val('kv_l70_kb')} KB: {val('l70_kv_heads')} shared",
         cc.kv_kb(unshared)),
        ("PaLM 540B", cc.kv_kb(cc.PALM), "memory", f"{val('kv_palm_kb')} KB: {cc.PALM.kv_heads} shared", None),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 2.45))
    n = len(rows)
    for i, (label, v, ck, txt, ghost) in enumerate(rows):
        y = n - 1 - i
        if ghost:
            ax.barh(y, ghost - 30, left=30, color="white", edgecolor=C["danger"],
                    ls="--", lw=1.4, height=0.62)
            ax.text(ghost * 1.1, y, f"{val('kv_l70_unshared_mb')} MB\nunshared", va="center", fontsize=10.5,
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
    hbm, pcie = cc.m(cc.GATHER.device_rate, "1/s"), cc.m(cc.GATHER.host_rate, "1/s")
    fig, ax = plt.subplots(figsize=(2.6, 2.55))
    for x, v, ck, txt in [(0, hbm, "memory", f"{val('rec_hbm_m')}M"),
                          (1, pcie, "network", f"{val('rec_pcie_m')}M")]:
        ax.bar(x, v, color=C[ck], width=0.62)
        ax.text(x, v * 1.4, txt, ha="center", va="bottom", fontsize=12.5,
                fontweight="bold", color=C[ck])
    ax.text(1, 3.2e7, f"{val('rec_ratio')}× fewer", ha="center", fontsize=12.5,
            fontweight="bold", color=C["danger"])
    ax.set_yscale("log")
    ax.set_ylim(1e6, 3e9)
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"HBM\n{val('h100_tbs')} TB/s", f"PCIe Gen5\n{val('pcie_gbs')} GB/s"])
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
    dev = num("a100_gb")                   # the labeled capacity the estimate divides by
    rows = [("One table", num("tbl_gb"), "memory", f"{val('tbl_gb')} GB"),
            ("Two tables", num("tbl2_gb"), "danger", f"{val('tbl2_gb')} GB")]
    for i, (label, v, ck, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=C[ck], height=0.6)
        ax.text(-2, y, label, ha="right", va="center", fontsize=12,
                fontweight="bold", color=C["text"])
        if v < dev:
            ax.text(v - 2, y, txt, va="center", ha="right", fontsize=12,
                    fontweight="bold", color="white")
        else:
            ax.text(v + 3, y, txt, va="center", fontsize=12, fontweight="bold",
                    color=C[ck])
    ax.axvline(dev, color=C["danger"], lw=2, ls="--")
    ax.text(dev - 2, 1.62, f"{val('a100_gb')} GB A100", ha="right", fontsize=11, color=C["danger"],
            fontweight="bold")
    ax.set_xlim(0, 150)
    ax.set_ylim(-0.5, 1.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xlabel(f"GB ({val('tbl_rows_m')}M rows, {val('rec_width')}-wide, FP32)")
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    save(fig, out / "capacity.svg",
         dict(left=0.27, right=0.97, bottom=0.27, top=0.97))


# --- Synthesis: one 1M-token request (derived, illustrative shape) ---------
def million_request(step):
    weights = cc.m(cc.KV_L8_1M.weights, "GB")
    kv = cc.m(cc.KV_L8_1M.total, "GB")
    scores = cc.m(cc.SCORES_1M.layer_bytes, "GB")   # per head per layer
    rows = [("Weights, BF16", weights, "memory", f"{val('mil_weights_gb')} GB"),
            (f"KV cache, {k_label(cc.KV_L8_1M.tokens)} tokens", kv, "compute", f"{val('mil_kv_gb')} GB"),
            ("Scores if stored", scores, "danger", f"{val('mil_scores_tb')} TB per head, per layer")]
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
            elif v < num("dev_gb"):
                ax.text(v / 1.12, y, txt, va="center", ha="right", fontsize=12,
                        fontweight="bold", color="white")
            else:
                ax.text(v * 1.15, y, txt, va="center", fontsize=12,
                        fontweight="bold", color=C[ck])
        ax.text(0.92, y, label, ha="right", va="center", fontsize=12,
                fontweight="bold", color=C["text"] if shown else "white")
    ax.axvline(num("dev_gb"), color=C["danger"], lw=2, ls=":")
    ax.text(num("dev_gb") * 1.1, n - 0.42, f"{val('dev_gb')} GB device", ha="left", va="bottom", fontsize=10.5,
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
    row = cc.row                              # DeepSeek-R1-0528, vol3/04_context_engineering/data/fiction_livebench_by_length.csv
    cols = [c for c in load_series("vol3/04_context_engineering/data/fiction_livebench_by_length.csv").columns if c.startswith("score_")
            and row[c] is not None and cc.col_tokens(cc.short_col) <= cc.col_tokens(c) <= cc.col_tokens(cc.long_col)]
    xs = [cc.col_tokens(c) for c in cols]
    ys = [row[c] for c in cols]
    fig, ax = plt.subplots(figsize=(2.9, 2.2))
    ax.plot(xs, ys, color=C["memory"], lw=3, marker="o", ms=6)
    ax.text(xs[0], ys[0] + 0.04, val("fl_short"), ha="left", va="bottom",
            fontsize=11, fontweight="bold", color=C["memory"])
    ax.text(xs[-1], ys[-1] - 0.05, val("fl_long"), ha="right", va="top",
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



# --- Quadratic: FlashAttention measured (Dao et al. 2022, Tables 20 and 21) --
def flash_measured():
    rows = cc.FLASH.entries
    seq = [r["seq_len"] for r in rows]

    def col(name, scale=1):
        return [r[name] / scale for r in rows if r[name] is not None]

    def per_doubling(name):
        last, prev = rows[-1][name], rows[-2][name]
        return f"×{last / prev:.0f}"

    fig, axs = plt.subplots(1, 2, figsize=(5.6, 1.4))
    panels = [(axs[0], col("pytorch_runtime_ms"), col("flashattention_runtime_ms"), "Runtime (ms)",
               f"FlashAttention:\n{per_doubling('flashattention_runtime_ms')} per doubling", (0.1, 3e4), 0.45),
              (axs[1], col("pytorch_memory_mb", 1e3), col("flashattention_memory_mb", 1e3), "Memory (GB)",
               f"FlashAttention:\n{per_doubling('flashattention_memory_mb')} per doubling", (0.01, 3000), 0.045)]
    for ax, pt, fa, ylab, lab, ylim, ylab_y in panels:
        ax.plot(seq[:len(pt)], pt, color=C["danger"], lw=2.8, marker="o", ms=4.5)
        ax.plot(seq, fa, color=C["memory"], lw=2.8, marker="o", ms=4.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(100, 9e4)
        ax.set_ylim(*ylim)
        ax.set_xticks([128, 1024, 8192, 65536])
        ax.set_xticklabels(["128 tokens", "1K", "8K", "64K"])
        ax.minorticks_off()
        ax.set_ylabel(ylab)
        ax.grid(color=C["grid"], lw=0.6)
        ax.text(8.5e4, ylab_y, lab, ha="right", va="center", fontsize=10.5,
                fontweight="bold", color=C["memory"], linespacing=1.0)
    axs[0].set_yticks([1, 100, 1e4])
    axs[0].set_yticklabels(["1", "100", "10⁴"])
    axs[1].set_yticks([0.01, 1, 100])
    axs[1].set_yticklabels(["0.01", "1", "100"])
    axs[0].text(140, 2.5e3, "PyTorch:\nstores scores", color=C["danger"],
                fontsize=10.5, fontweight="bold", va="center", linespacing=1.0)
    axs[1].text(140, 300, f"PyTorch: out of\nmemory after {ki_label(seq[len(col('pytorch_memory_mb')) - 1])}", color=C["danger"],
                fontsize=10.5, fontweight="bold", va="center", linespacing=1.0)
    save(fig, out / "flash_measured.svg",
         dict(left=0.12, right=0.985, bottom=0.17, top=0.97, wspace=0.42))


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
flash_measured()


print("figures written")
