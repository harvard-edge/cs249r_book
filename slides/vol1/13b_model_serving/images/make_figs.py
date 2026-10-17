#!/usr/bin/env python3
"""Figures for Lecture 13b (Model Serving, Part 2: Serving a Stream of Tokens).

Every value is sourced. "MS" is the rendered chapter, Model Serving. Values
computed here are labeled "derived" on the figure or its slide.

Moved from Part 1 (Lecture 13) and regenerated from a copy of its code:
ttft (extended with the A100 row and the book's targets), decode_bytes,
kv_budget, vllm (split into two single-panel figures), joules_token, and
Part 1's hook_cost bars as the recall thumbnail.

  - MLPerf Inference v5.0, closed, datacenter, summary_results.json
    (mlcommons/inference_results_v5.0), 5.0-0020 Dell PowerEdge XE9680,
    8x H100-SXM-80GB: llama2-70b-99 Offline 31,216.8 tokens/s, Server
    30,903.6 (TTFT/TPOT 2,000/200 ms); llama2-70b-interactive-99 Server
    12,979.4 (450/40 ms). Bounds from the v5.0-era inference rules.
    Ratios and Little's law bounds derived: 0.990, 0.420; 6,181 and 519.
  - Splitwise (Patel et al., ISCA 2024, Table IV): Llama 2 70B, P50, no
    batching, conversation trace: DGX-H100 TTFT 84 ms, TBT 28 ms, E2E
    3,387 ms; DGX-A100 155, 40, 4,957 ms. Decode share derived (97 percent).
  - Book targets (MS Systems Perspective 13.5, illustrative): TTFT < 500 ms,
    TPOT < 50 ms.
  - Prefill and decode (MS Sec. 13.9.2, Sec. 13.10.3): GPT-2 XL 512-token
    prefill 3.9 ms, decode floor 0.9 ms, eager 576 kernels 8.6 ms; Llama 3
    8B 1,000-token prefill 41.9 ms, INT4 floor 1.2 ms, modeled TPOT 1.57 ms,
    TPOT target 20 ms.
  - Speculative decoding (MS Sec. 13.9.4, equation 15): draft 0.30 ms,
    verify 0.90 ms, k = 4, acceptance 0.7 (assumed), 2.8 tokens per round.
  - GPT-2 XL decode bytes (MS Sec. 13.9.2): weights 3.0 GB, cache 0.3146 GB
    per full 1,024-token context. Batch curve derived: step time =
    (3.0 + 0.3146 B) GB / 3.35 TB/s; memory full at B = 263 (85.9 GB).
  - Continuous batching schedule: illustrative lengths, one token per step,
    prefill omitted.
  - Stall timeline, derived: Llama 3 8B INT4 with 32 decodes of 1,256 tokens:
    (4 GB + 32 x 1,256 x 131,072 B) / 3.35 TB/s = 2.8 ms; prefill 41.9 ms
    per 1,000 tokens (MS Sec. 13.10.3), linear in tokens.
  - KV budget, 70B class (MS figure 8 variant): 327,680 B per token assumed
    (80 layers, 8 KV heads of 128, FP16); budget 50.6 GB.
  - vLLM (Kwon et al., SOSP 2023): token-state share of KV memory, Orca (Max)
    20.4, (Pow2) 26.8, (Oracle) 38.2, vLLM 96.3 percent (Fig. 2); average
    batched requests, OPT-13B, ShareGPT: Orca (Oracle) 13.62, vLLM 30.42
    (Fig. 13a; shown on a card). Block table after Fig. 6 (block size 4 here; vLLM default 16).
  - Paged budget, derived: Llama 3 8B FP16 on one H100, 69.8 GB free;
    reserve 8,192 tokens = 1.07 GB, 65; page 1,256 tokens in 79 blocks of 16
    = 165.7 MB, 421 (MS Sec. 13.9.5, Sec. 13.10.3 workload).
  - Prefix sharing, derived: 2,000 tokens x 131,072 B = 262 MB per copy
    (MS Sec. 13.9.5).
  - DistServe (Zhong et al., OSDI 2024, Fig. 1 and Sec. 1): 13B model, one
    A100, 512 in, 64 out: colocated 1.6, prefill only 5.6, decode only 10
    requests/s per GPU; 2 prefill + 1 decode, 3.3 per GPU. Sec. 3.3: 1.13 GB
    per 512-token OPT-66B request, 11.3 GB/s at 10 requests/s; NVLink
    600 GB/s.
  - Joules per token, derived: 1,000 W node (MS Napkin Math 13.5) over
    1,010 and 8,200 tokens/s.

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

C = sp.C
plt = sp.plt
out = Path(__file__).parent


def tag(ax, text, x=0.0, y=-0.30):
    """A small gray provenance tag under the axes (modeled, derived ...)."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9,
            color=C["neutral"], ha="left", va="top")


def hide_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ------------------------------------------------------------------ hook
HOOK = [("No deadline\n(Offline)", 31216.8, "neutral", "31,217"),
        ("TTFT 2 s,\nTPOT 200 ms", 30903.6, "compute", "30,904  (99%)"),
        ("TTFT 450 ms,\nTPOT 40 ms", 12979.4, "danger", "12,979  (42%)")]


def hook_fig(path, k, size=(5.8, 2.6), note=True, fs=12):
    fig, ax = plt.subplots(figsize=size)
    for i, (lab, v, ck, txt) in enumerate(HOOK):
        y = 2 - i
        shown = i < k if i < 2 else k >= 4
        ax.barh(y, v / 1000, color=C[ck], height=0.6, alpha=1.0 if shown else 0.0)
        ax.text(-0.4, y, lab, ha="right", va="center", fontsize=fs, fontweight="bold",
                alpha=1.0 if (i < k or (i == 2 and k >= 3)) else 0.0)
        if i == 2 and k == 3:
            ax.text(0.6, y, "???", va="center", fontsize=fs + 2, fontweight="bold",
                    color=C["danger"])
        ax.text(v / 1000 + 0.5, y, txt, va="center", fontsize=fs, fontweight="bold",
                color=C[ck], alpha=1.0 if shown else 0.0)
    if note:
        ax.text(39, -0.75, "same chips, same model; 5× tighter TPOT: 2.4× fewer tokens",
                ha="right", va="center", fontsize=11, color=C["danger"],
                fontweight="bold", alpha=1.0 if k >= 4 else 0.0)
    ax.set_yticks([])
    ax.set_xlim(0, 40)
    ax.set_ylim(-1.1 if note else -0.5, 2.5)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Thousand tokens/s, Llama 2 70B on one 8× H100 server")
    sp.save(fig, path)


for k in (1, 2, 3, 4):
    hook_fig(out / f"hook_step{k}.svg", k)
hook_fig(out / "hook_thumb.svg", 4, size=(3.6, 2.0), note=False, fs=10)

# ------------------------------------------- recall thumbnail (Part 1 hook)
sp.hbars(out / "recall_cost.svg", [
    ("Batch 1", 3.92, "danger", "\\$3.92"),
    ("Batch 8", 0.52, "compute", "\\$0.52"),
], "", log=False, size=(4.0, 1.4),
    xlabel="ResNet-50, \\$ per million requests (modeled)")

# ---------------------------------------------------------- TTFT and TPOT
for k in (1, 2, 3, 4):
    fig, ax = plt.subplots(figsize=(5.9, 2.6))
    rows = [("DGX-H100", 1.3, 84, 28, 3387), ("DGX-A100", 0.0, 155, 40, 4957)]
    for name, y, ttft, tbt, e2e in rows:
        a = 1.0 if (y > 0.5 or k >= 3) else 0.0
        ax.barh(y, ttft, color=C["compute"], height=0.42, alpha=a)
        for t in np.arange(ttft + tbt, e2e, tbt):
            ax.plot([t, t], [y - 0.2, y + 0.2], color=C["memory"], lw=1.1, alpha=a)
        ax.text(-60, y, name, ha="right", va="center", fontsize=12,
                fontweight="bold", alpha=a)
        ax.text(e2e + 50, y, f"{e2e:,} ms", va="center", fontsize=11,
                fontweight="bold", alpha=a)
        ax.text(600, y + 0.3, f"first token {ttft} ms, then one every {tbt} ms",
                fontsize=10.5, color=C["memory"], fontweight="bold", alpha=a)
    if k >= 2:
        ax.text(600, 0.82, "decode: 97% of the request (derived)", ha="left",
                va="center", fontsize=10.5, color=C["memory"], fontweight="bold")
    if k >= 4:
        ax.axvline(500, color=C["danger"], lw=2, ls="--")
        ax.text(600, -0.62, "book target: TTFT < 500 ms;  TPOT < 50 ms (20 tokens/s)",
                fontsize=10.5, color=C["danger"], fontweight="bold", va="center")
    ax.set_xlim(0, 5600)
    ax.set_ylim(-0.85, 1.85)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Milliseconds, one chat request (Llama 2 70B, median, no batching)")
    sp.save(fig, out / f"ttft_step{k}.svg")

# ------------------------------------------------- prefill against decode
for k in (1, 2, 3, 4):
    fig = plt.figure(figsize=(5.4, 2.7))
    ax = fig.add_axes([0.20, 0.50, 0.30, 0.38])
    bx = fig.add_axes([0.67, 0.50, 0.30, 0.38])
    cx = fig.add_axes([0.20, 0.08, 0.77, 0.14])
    for i, (lab, v) in enumerate((("GPT-2 XL,\n512 tokens", 3.9),
                                  ("Llama 3 8B,\n1,000 tokens", 41.9))):
        y = 1 - i
        ax.barh(y, v, color=C["compute"], height=0.6)
        ax.text(v + 1.5 if v < 30 else v - 1.5, y, f"{v} ms", va="center",
                ha="left" if v < 30 else "right", fontsize=11, fontweight="bold",
                color=C["compute"] if v < 30 else "white")
        ax.text(-2, y, lab, ha="right", va="center", fontsize=10.5, fontweight="bold")
    ax.set_xlim(0, 50)
    ax.set_title("Prefill: TFLOP ÷ (40% of peak)", fontsize=11, loc="left",
                 fontweight="bold", color=C["compute"])
    ax.set_xlabel("ms (sets TTFT)", fontsize=10)
    a2 = 1.0 if k >= 2 else 0.0
    drows = [("GPT-2 XL", 0.9, "memory", "0.9 ms"),
             ("Llama 3 8B\nINT4", 1.57, "memory", "1.2 floor, 1.57 ms"),
             ("GPT-2 XL,\neager", 8.6, "neutral", "8.6 ms")]
    for i, (lab, v, ck, txt) in enumerate(drows):
        y = 2 - i
        a = a2 if i < 2 else (1.0 if k >= 3 else 0.0)
        bx.barh(y, v, color=C[ck], height=0.6, alpha=a)
        bx.text(v + 0.3 if v < 5 else v - 0.3, y, txt, va="center",
                ha="left" if v < 5 else "right", fontsize=10.5, fontweight="bold",
                color=C[ck] if v < 5 else "white", alpha=a)
        bx.text(-0.4, y, lab, ha="right", va="center", fontsize=10.5,
                fontweight="bold", alpha=a)
    bx.set_xlim(0, 10)
    bx.set_title("Decode step: GB ÷ 3.35 TB/s", fontsize=11, loc="left",
                 fontweight="bold", color=C["memory"], alpha=a2)
    bx.set_xlabel("ms (sets TPOT)", fontsize=10, alpha=a2)
    if k < 2:
        bx.spines["bottom"].set_alpha(0.0)
        bx.tick_params(axis="x", colors="white")
    for a_ in (ax, bx):
        a_.set_yticks([])
        a_.spines["left"].set_visible(False)
        a_.set_ylim(-0.5, 2.5 if a_ is bx else 1.5)
    a4 = 1.0 if k >= 4 else 0.0
    cx.barh(0, 1.57, color=C["memory"], height=0.8, alpha=a4)
    cx.barh(0, 20 - 1.57, left=1.57, color=C["data_l"], edgecolor=C["data"],
            hatch="//", height=0.8, lw=0, alpha=a4)
    cx.text(10.8, 0, "18.4 ms: room to share this weight read (derived)", ha="center",
            va="center", fontsize=10.5, fontweight="bold", color=C["data"], alpha=a4,
            bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=a4))
    cx.text(-0.3, 0, "Llama 3 8B step\nin a 20 ms TPOT", ha="right", va="center",
            fontsize=10.5, fontweight="bold", alpha=a4)
    cx.set_xlim(0, 20)
    cx.set_ylim(-0.6, 0.6)
    cx.set_yticks([])
    cx.set_xticks([0, 5, 10, 15, 20])
    cx.spines["left"].set_visible(False)
    cx.tick_params(axis="x", labelsize=10, colors=C["text"] if k >= 4 else "white")
    cx.spines["bottom"].set_alpha(a4)
    sp.save(fig, out / f"phases_step{k}.svg")

# ------------------------------------------------- speculative decoding
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.9, 2.0))
    s = 1.5  # drawing units per ms
    ax.add_patch(Rectangle((0, 1.0), 0.30 * s, 0.5, color=C["data"]))
    ax.text(0.15 * s, 1.25, "draft", ha="center", va="center", fontsize=10.5,
            color="white", fontweight="bold")
    ax.text(0.0, 1.65, "draft proposes 4 tokens (0.30 ms), then the target checks all 5 positions in one pass",
            ha="left", fontsize=10.5, color=C["text"], fontweight="bold")
    ax.add_patch(Rectangle((0.30 * s, 1.0), 0.90 * s, 0.5, color=C["memory"]))
    ax.text(0.75 * s, 1.25, "target: 0.90 ms", ha="center", va="center", fontsize=10.5,
            color="white", fontweight="bold")
    ax.text(1.20 * s + 0.05, 1.25, "1.20 ms per round", va="center", fontsize=10.5,
            fontweight="bold")
    a = 1.0 if k >= 2 else 0.0
    marks = [("t1", "data", "kept"), ("t2", "data", "kept"), ("t3", "danger", "rejected"),
             ("t4", "neutral", "dropped"), ("new", "memory", "target's")]
    for j, (lab, ck, word) in enumerate(marks):
        x = j * 0.36
        ax.add_patch(Rectangle((x, 0.05), 0.30, 0.45, facecolor=C[ck + "_l"],
                               edgecolor=C[ck], lw=1.5, alpha=a))
        ax.text(x + 0.15, 0.275, lab, ha="center", va="center", fontsize=10,
                fontweight="bold", color=C[ck], alpha=a)
        ax.text(x + 0.15, -0.14, word, ha="center", va="center", fontsize=9.5,
                color=C[ck], alpha=a)
    ax.text(1.85, 0.275, "one example round: 3 tokens", va="center", fontsize=10.5,
            fontweight="bold", alpha=a)
    ax.set_xlim(0, 2.9)
    ax.set_ylim(-0.3, 1.9)
    hide_axis(ax)
    sp.save(fig, out / f"spec_step{k}.svg")

# ------------------------------------------------------- decode bytes
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    for i, (b, cache, ms, rate) in enumerate(((1, 0.31, "1.0 ms", "1,010 tokens/s"),
                                              (32, 10.07, "3.9 ms", "8,200 tokens/s"))):
        y = 1 - i
        a = 1.0 if (k == 2 or b == 1) else 0.0
        ax.barh(y, 3.0, color=C["memory"], height=0.55, alpha=a)
        ax.barh(y, cache, left=3.0, color=C["data"], height=0.55, alpha=a)
        ax.text(3.0 + cache + 0.2, y, f"{3.0 + cache:.1f} GB: {ms}, {rate}", va="center",
                fontsize=11, fontweight="bold", color=C["text"], alpha=a)
        ax.text(-0.15, y, f"Batch {b}", ha="right", va="center", fontsize=12,
                fontweight="bold", alpha=a)
    ax.text(1.5, 1.45, "weights 3 GB", ha="center", color=C["memory"], fontsize=11,
            fontweight="bold")
    ax.text(8.0, -0.55, "KV cache, 0.31 GB per sequence", ha="center", va="top",
            color=C["data"], fontsize=11, fontweight="bold", alpha=1.0 if k == 2 else 0.0)
    ax.set_xlim(0, 22)
    ax.set_ylim(-0.9, 1.7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Bytes read per decode step (GB; GPT-2 XL, 1,024-token context, modeled)")
    sp.save(fig, out / f"decode_bytes_step{k}.svg")

# ------------------------------------------------- batch curve and TPOT cap


def step_ms(b):
    return (3.0 + 0.3146 * b) / 3.35


for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.9, 2.7))
    top = {1: 32, 2: 203, 3: 263}[k]
    bs = np.arange(1, top + 1)
    ax.plot(bs / step_ms(bs) * 1000, step_ms(bs), color=C["memory"], lw=3)
    pts = [(1, "batch 1"), (8, "8"), (32, "32"), (64, "64"), (128, "128"),
           (203, "203"), (263, "")]
    for b, lab in pts:
        if b > top:
            continue
        x, y = b / step_ms(b) * 1000, step_ms(b)
        derived = b > 32
        ax.plot([x], [y], "o", ms=7, color=C["neutral"] if derived else C["memory"],
                mec="white", zorder=5)
        ax.annotate(lab, (x, y), xytext=(-8, 3), textcoords="offset points",
                    ha="right", fontsize=10.5, fontweight="bold",
                    color=C["neutral"] if derived else C["memory"])
    ax.text(150, 3.9 + 0.2, "", fontsize=1)
    if k >= 2:
        ax.axhline(20, color=C["danger"], lw=2, ls="--")
        ax.text(200, 21, "TPOT target 20 ms: at most batch 203", fontsize=10.5,
                color=C["danger"], fontweight="bold")
        ax.axvline(10650, color=C["neutral"], lw=1.5, ls=":")
        ax.text(10500, 29.6, "ceiling 10,650 tokens/s:\ncache reads dominate",
                ha="right", va="top", fontsize=10, color=C["neutral"], fontweight="bold")
    if k >= 3:
        ax.text(10750, 25.0, "263: memory full", ha="left", fontsize=10.5,
                color=C["danger"], fontweight="bold")
    ax.set_xlim(0, 12000)
    ax.set_ylim(0, 30)
    ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000, 12000])
    ax.set_xticklabels(["0", "2K", "4K", "6K", "8K", "10K", "12K"])
    ax.set_xlabel("Aggregate tokens/s")
    ax.set_ylabel("TPOT (ms)")
    tag(ax, "GPT-2 XL on one H100, full 1,024-token contexts. Book: batch 1 and 32; "
        "gray points derived (bandwidth floor only).", y=-0.27)
    sp.save(fig, out / f"frontier_step{k}.svg")

# ------------------------------------------------ continuous batching grid
REQ1 = [("A", 10), ("B", 100), ("C", 40), ("D", 60)]
REQ2 = {"E": 30, "F": 50, "G": 20, "H": 40}


def grid(ax, title):
    for s in range(4):
        ax.axhspan(3 - s - 0.42, 3 - s + 0.42, xmin=0, xmax=1, color="#F4F4F4", lw=0, zorder=0)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(["slot 1", "slot 2", "slot 3", "slot 4"], fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 150)
    ax.set_ylim(-0.6, 3.6)
    ax.set_xticks([0, 10, 40, 60, 100, 150])
    ax.tick_params(axis="x", labelsize=9.5)
    ax.set_xlabel("Decode step (lengths illustrative; one token per step, prefill omitted)",
                  fontsize=10)
    ax.set_title(title, fontsize=11.5, loc="left", fontweight="bold")


def bar(ax, slot, start, length, name, ck):
    ax.barh(3 - slot, length, left=start, height=0.7, color=C[ck], edgecolor="white", lw=1.5)
    ax.text(start + length / 2, 3 - slot, f"{name} {length}" if length >= 18 else name,
            ha="center", va="center", fontsize=10, color="white", fontweight="bold")


def idle(ax, slot, start, length):
    ax.barh(3 - slot, length, left=start, height=0.7, color=C["danger_l"],
            edgecolor=C["danger"], hatch="//", lw=0)


def summary(ax, text, color):
    ax.text(1.0, -0.33, text, ha="right", va="top", fontsize=11, fontweight="bold",
            color=color, transform=ax.transAxes)


for k in range(1, 8):
    fig, ax = plt.subplots(figsize=(5.9, 2.7))
    if k <= 3:
        grid(ax, "Request-level: the batch is fixed until its longest member ends")
        for s, (n, L) in enumerate(REQ1):
            bar(ax, s, 0, L, n, "memory")
        if k == 1:
            idle(ax, 0, 10, 90)
        if k >= 2:
            for s, (n, L) in enumerate(REQ1):
                if L < 100:
                    idle(ax, s, L, 100 - L)
            ax.text(55, 3, "210 of 400 slot-steps used", ha="center", va="center",
                    fontsize=10.5, color=C["danger"], fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        if k == 3:
            for s, n in enumerate("EFGH"):
                bar(ax, s, 100, REQ2[n], n, "data")
            summary(ax, "done at step 150; 350 of 600 slot-steps used (58%)", C["danger"])
        else:
            ax.text(125, 1.5, "waiting:\nE 30, F 50,\nG 20, H 40", ha="center",
                    va="center", fontsize=10.5, color=C["data"], fontweight="bold")
    else:
        grid(ax, "Continuous: a freed slot takes the next request at the next step")
        for s, (n, L) in enumerate(REQ1):
            bar(ax, s, 0, L, n, "memory")
        if k == 4:
            ax.text(125, 1.5, "waiting:\nE 30, F 50,\nG 20, H 40", ha="center",
                    va="center", fontsize=10.5, color=C["data"], fontweight="bold")
        if k >= 5:
            bar(ax, 0, 10, 30, "E", "data")
        if k >= 6:
            bar(ax, 2, 40, 50, "F", "data")
            bar(ax, 0, 40, 20, "G", "data")
            bar(ax, 0, 60, 40, "H", "data")
            idle(ax, 2, 90, 10)
            idle(ax, 3, 60, 40)
        if k >= 7:
            ax.axvspan(100, 150, color="#FFFFFF", alpha=0.0)
            summary(ax, "done at step 100; 350 of 400 slot-steps used (87.5%)", C["data"])
    sp.save(fig, out / f"cb_step{k}.svg")

# ---------------------------------------------------------- stall timeline
DEC = (4e9 + 32 * 1256 * 131072) / 3.35e12 * 1000   # 2.77 ms per decode step
PRE = 41.9                                            # ms per 1,000-token prefill
for k in (1, 2, 3):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.9, 2.6), sharey=True,
                                 gridspec_kw={"wspace": 0.12})
    left = [DEC] * 4 + [DEC + PRE] + [DEC] * 4
    for i, v in enumerate(left):
        tall = i == 4
        a = 1.0 if (not tall or k >= 2) else 0.0
        ax.bar(i, v, color=C["compute"] if tall else C["memory"], width=0.75, alpha=a)
    ax.text(4, DEC + PRE + 1.5, "44.7 ms", ha="center", fontsize=11, fontweight="bold",
            color=C["compute"], alpha=1.0 if k >= 2 else 0.0)
    ax.set_title("1,000-token prefill in one step", fontsize=11, loc="left",
                 fontweight="bold")
    right = [DEC] * 2 + [DEC + PRE / 4] * 4 + [DEC] * 3
    a3 = 1.0 if k >= 3 else 0.0
    for i, v in enumerate(right):
        chunk = 2 <= i <= 5
        bx.bar(i, v, color=C["compute"] if chunk else C["memory"], width=0.75, alpha=a3)
    bx.text(4.5, 28, "4 chunks of 250 tokens:\n13.2 ms each", ha="center",
            fontsize=10.5, fontweight="bold", color=C["compute"], alpha=a3)
    bx.set_title("Split into four chunks", fontsize=11, loc="left", fontweight="bold",
                 alpha=a3)
    for a_, show in ((ax, True), (bx, k >= 3)):
        a_.axhline(20, color=C["danger"], lw=2, ls="--", alpha=1.0 if show else 0.0)
        a_.set_xticks([])
        a_.set_xlabel("decode steps", fontsize=10, alpha=1.0 if show else 0.0)
        a_.set_ylim(0, 52)
        if not show:
            a_.spines["bottom"].set_alpha(0.0)
    ax.text(8.6, 21, "TPOT 20 ms", ha="right", fontsize=10.5, color=C["danger"],
            fontweight="bold")
    ax.text(0, DEC + 1.5, "2.8 ms", ha="center", fontsize=10.5, color=C["memory"],
            fontweight="bold")
    ax.set_ylabel("Step time (ms)\n= wait for next token")
    tag(ax, "Derived: Llama 3 8B INT4, one H100, 32 running decodes; prefill at 40% of "
        "peak, linear in tokens.", y=-0.16)
    sp.save(fig, out / f"stall_step{k}.svg")

# ------------------------------------------- KV budget, 70B class (variant)
ctx = np.linspace(0, 32768, 200)
fig, ax = plt.subplots(figsize=(5.8, 2.9))
ax.axhspan(50.6, 125, color=C["danger_l"], alpha=0.6, lw=0)
ax.axhline(50.6, color=C["danger"], lw=2, ls="--")
ax.text(32000, 53, "budget after 4-bit weights: 50.6 GB", ha="right", va="bottom",
        fontsize=11, color=C["danger"], fontweight="bold")
for b, ck in ((1, "memory"), (4, "data"), (16, "compute"), (32, "network")):
    kv = b * ctx * 327680 / 1e9
    ax.plot(ctx, kv, color=C[ck], lw=3)
    if b > 4:
        xe = 118e9 / (b * 327680)
        lx, ly = xe * 0.97, b * xe * 327680 / 1e9 + 2
    else:
        lx = 24000 if b == 4 else 32000
        ly = b * lx * 327680 / 1e9 + 3
    ax.text(lx, ly, f"batch {b}", ha="right", fontsize=11, color=C[ck], fontweight="bold")
x32 = 50.6e9 / (32 * 327680)
ax.plot([x32], [50.6], "o", ms=9, color=C["network"], mec="white", mew=1.5, zorder=5)
ax.annotate("32 chats: full near 4,800 tokens", (x32, 50.6), xytext=(17500, 74),
            fontsize=11, color=C["network"], fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=C["network"]))
ax.set_xlim(0, 32768)
ax.set_ylim(0, 120)
ax.set_xticks([0, 8192, 16384, 24576, 32768])
ax.set_xticklabels(["0", "8K", "16K", "24K", "32K"])
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("KV cache (GB, FP16)")
tag(ax, "After Model Serving, figure 8: 70B-class model, grouped-query attention, one H100",
    y=-0.26)
sp.save(fig, out / "kv_budget.svg")

# ------------------------------------------------ fragmentation (after vLLM)
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.9, 1.9))
    y0, h = 0.0, 0.8

    def seg(x, w, fc, ec, hatch=None, label=None, tc="white", a=1.0):
        ax.add_patch(Rectangle((x, y0), w, h, facecolor=fc, edgecolor=ec, hatch=hatch,
                               lw=1.2, alpha=a))
        if label:
            ax.text(x + w / 2, y0 + h / 2, label, ha="center", va="center", fontsize=10,
                    fontweight="bold", color=tc, alpha=a)

    ax.add_patch(Rectangle((0, y0), 600, h, facecolor="white", edgecolor=C["neutral"], lw=1.2))
    seg(0, 70, C["data"], "white", label="A: 70")
    seg(70, 130, C["compute_l"], C["compute"], hatch="//")
    ax.text(135, -0.25, "reserved for A's maximum", ha="center", va="center", fontsize=10,
            color=C["compute"], fontweight="bold")
    a2 = 1.0 if k >= 2 else 0.0
    seg(300, 120, C["data"], "white", label="B: 120", a=a2)
    seg(420, 80, C["danger_l"], C["danger"], hatch="xx", a=a2)
    ax.text(460, -0.25, "internal: B stops at 120", ha="center", va="center", fontsize=10,
            color=C["danger"], fontweight="bold", alpha=a2)
    a3 = 1.0 if k >= 3 else 0.0
    ax.add_patch(Rectangle((200, y0), 100, h, facecolor=C["neutral_l"], edgecolor=C["neutral"],
                           lw=1.2, alpha=a3))
    ax.text(250, y0 + h / 2, "free 100", ha="center", va="center", fontsize=10,
            fontweight="bold", color=C["text"], alpha=a3)
    ax.add_patch(Rectangle((175, 1.3), 150, 0.5, facecolor=C["memory_l"],
                           edgecolor=C["memory"], lw=1.2, alpha=a3))
    ax.text(250, 1.55, "new request: 150", ha="center", va="center", fontsize=10,
            fontweight="bold", color=C["memory"], alpha=a3)
    ax.text(335, 1.55, "external: the hole is too small", va="center", fontsize=10,
            color=C["danger"], fontweight="bold", alpha=a3)
    ax.set_xlim(-5, 605)
    ax.set_ylim(-0.5, 2.0)
    hide_axis(ax)
    sp.save(fig, out / f"frag_step{k}.svg")

# ------------------------------------------------------------ vLLM measured
names = ["Orca (Max)", "Orca (Pow2)", "Orca (Oracle)", "vLLM"]
used = [20.4, 26.8, 38.2, 96.3]
fig, ax = plt.subplots(figsize=(5.8, 1.9))
for i, n in enumerate(names):
    y = 3 - i
    ck = "data" if n == "vLLM" else "neutral"
    ax.barh(y, used[i], color=C[ck], height=0.62)
    ax.text(used[i] + 1.5, y, f"{used[i]:g}%", va="center", fontsize=11,
            fontweight="bold", color=C[ck])
    ax.text(-1.5, y, n, ha="right", va="center", fontsize=11, fontweight="bold")
ax.set_xlim(0, 110)
ax.set_yticks([])
ax.set_ylim(-0.5, 3.5)
ax.spines["left"].set_visible(False)
ax.set_xlabel("Measured share of KV memory holding token states (percent)")
sp.save(fig, out / "vllm_used.svg")

# ---------------------------------------------------------- paging worked
A_TOK = ["Four", "score", "and", "seven", "years", "ago", "our", "fathers", "brought"]
B_TOK = ["It", "was", "the", "best", "of", "times"]
A_PHYS = [7, 1, 3]
B_PHYS = [5, 2]
CW, CH = 0.6, 0.32
for k in (1, 2, 3, 4):
    fig, ax = plt.subplots(figsize=(5.9, 3.0))
    na = {1: 7, 2: 8, 3: 9, 4: 9}[k]
    new_tok = {2: 7, 3: 8}.get(k, -1)
    ax.text(0.05, 3.55, "Request A (logical)", fontsize=10.5, fontweight="bold",
            color=C["memory"])
    for lb in range(3):
        y = 3.0 - lb * 0.5
        a = 1.0 if lb * 4 < na else 0.0
        ax.text(-0.02, y + CH / 2, f"{lb}", ha="right", va="center", fontsize=10, alpha=a)
        for j in range(4):
            t = lb * 4 + j
            ax.add_patch(Rectangle((0.05 + j * CW, y), CW, CH, facecolor="white",
                                   edgecolor=C["neutral"], lw=1, alpha=a))
            if t < na:
                ax.text(0.05 + j * CW + CW / 2, y + CH / 2, A_TOK[t], ha="center",
                        va="center", fontsize=8,
                        color=C["danger"] if t == new_tok else C["text"],
                        fontweight="bold" if t == new_tok else "normal")
    ax.text(2.6, 3.55, "Block table", fontsize=10.5, fontweight="bold")
    for lb in range(3):
        y = 3.0 - lb * 0.5
        a = 1.0 if lb * 4 < na else 0.0
        filled = max(0, min(4, na - lb * 4))
        changed = (k == 2 and lb == 1) or (k == 3 and lb == 2)
        ax.add_patch(Rectangle((2.6, y), 1.45, CH, facecolor=C["memory_l"],
                               edgecolor=C["memory"], lw=1, alpha=a))
        ax.text(3.325, y + CH / 2, f"{lb} → {A_PHYS[lb]}  ({filled} filled)", ha="center",
                va="center", fontsize=9, alpha=a,
                color=C["danger"] if changed else C["text"],
                fontweight="bold" if changed else "normal")
    ax.text(4.45, 3.55, "Physical blocks", fontsize=10.5, fontweight="bold")
    for pb in range(8):
        y = 3.05 - pb * 0.37
        ax.text(4.4, y + 0.14, f"{pb}", ha="right", va="center", fontsize=9.5)
        owner = None
        if pb in A_PHYS:
            owner = ("A", A_PHYS.index(pb))
        elif k >= 4 and pb in B_PHYS:
            owner = ("B", B_PHYS.index(pb))
        for j in range(4):
            fc, ec, word = "white", C["neutral"], None
            if owner:
                toks = A_TOK[:na] if owner[0] == "A" else B_TOK
                t = owner[1] * 4 + j
                if t < len(toks):
                    word = toks[t]
                    fc = C["memory_l"] if owner[0] == "A" else C["data_l"]
                    ec = C["memory"] if owner[0] == "A" else C["data"]
            ax.add_patch(Rectangle((4.45 + j * CW, y), CW, 0.28, facecolor=fc,
                                   edgecolor=ec, lw=0.8))
            if word:
                ax.text(4.45 + j * CW + CW / 2, y + 0.14, word, ha="center", va="center",
                        fontsize=8)
    captions = {1: "7-token prompt: 2 blocks, 1 slot free",
                2: "Step 1: 'fathers' fills the free slot",
                3: "Step 2: a new block, anywhere free",
                4: "Request B shares the same pool"}
    ax.text(0.05, 1.3, captions[k], fontsize=10.5, fontweight="bold")
    ax.text(0.05, 0.9, "Request B: logical 0 → 5, 1 → 2", fontsize=10.5,
            color=C["data"], fontweight="bold", alpha=1.0 if k >= 4 else 0.0)
    ax.text(0.05, 0.5, "Block size 4 for legibility (vLLM default: 16)", fontsize=9,
            color=C["neutral"])
    ax.set_xlim(-0.2, 6.95)
    ax.set_ylim(0.3, 3.8)
    hide_axis(ax)
    sp.save(fig, out / f"paging_step{k}.svg")

# ------------------------------------------------------ paged budget
sp.hbars(out / "paged_budget.svg", [
    ("Reserve 8,192 tokens", 65, "neutral", "65 conversations"),
    ("Page 1,256 tokens", 421, "data", "about 421"),
], "", log=False, size=sp.SHORT,
    xlabel="Llama 3 8B, FP16 weights, one H100 (derived, before runtime workspace)")

# ------------------------------------------------------ prefix sharing
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.9, 2.4))
    for j in range(5):
        ax.add_patch(Rectangle((0.2 + j * 0.42, 1.6), 0.38, 0.5, facecolor=C["network_l"],
                               edgecolor=C["network"], lw=1.2))
    ax.text(1.24, 2.35, "system prompt: 2,000 tokens, 262 MB, stored once", ha="center",
            fontsize=10.5, color=C["network"], fontweight="bold")
    for i in range(3):
        x = 0.2 + i * 0.8
        ax.add_patch(Rectangle((x, 0.2), 0.6, 0.45, facecolor=C["memory_l"],
                               edgecolor=C["memory"], lw=1.2))
        ax.text(x + 0.3, 0.42, f"chat {i + 1}" if i < 2 else "chat N", ha="center",
                va="center", fontsize=10)
        ax.add_patch(FancyArrowPatch((x + 0.3, 0.68), (x + 0.3 + (1.2 - x - 0.3) * 0.5, 1.55),
                                     arrowstyle="-|>", mutation_scale=10,
                                     color=C["network"], lw=1.3))
    ax.text(1.3, -0.05, "each block table points at the shared blocks", ha="center",
            fontsize=9.5, color=C["neutral"])
    a = 1.0 if k >= 2 else 0.0
    bars = [("100 copies", 26.2, "danger", "26.2 GB"), ("Shared", 0.262, "data", "0.26 GB")]
    for i, (lab, v, ck, txt) in enumerate(bars):
        y = 1.6 - i * 0.7
        ax.add_patch(Rectangle((3.4, y), v / 26.2 * 1.9, 0.45, color=C[ck], alpha=a))
        ax.text(3.35, y + 0.22, lab, ha="right", va="center", fontsize=10.5,
                fontweight="bold", alpha=a)
        ax.text(3.4 + v / 26.2 * 1.9 + 0.05, y + 0.22, txt, va="center", fontsize=10.5,
                fontweight="bold", color=C[ck], alpha=a)
    ax.text(4.3, 0.35, "100 chats (derived)", ha="center", fontsize=9.5,
            color=C["neutral"], alpha=a)
    ax.set_xlim(0, 6.0)
    ax.set_ylim(-0.2, 2.6)
    hide_axis(ax)
    sp.save(fig, out / f"share_step{k}.svg")

# ------------------------------------------------------------ DistServe
sp.hbars(out / "distserve_bars.svg", [
    ("Both phases", 1.6, "neutral", "1.6"),
    ("Prefill only", 5.6, "compute", "5.6"),
    ("Decode only", 10, "memory", "10"),
], "", log=False, value_fmt="{:g}", size=(5.8, 2.0),
    xlabel="Requests/s per A100 meeting both deadlines (13B model, measured)")

fig, ax = plt.subplots(figsize=(4.4, 2.3))
for i, (lab, ck) in enumerate((("prefill", "compute"), ("prefill", "compute"))):
    ax.add_patch(Rectangle((0.1, 1.3 - i * 1.0), 1.2, 0.7, facecolor=C[ck + "_l"],
                           edgecolor=C[ck], lw=1.8))
    ax.text(0.7, 1.65 - i * 1.0, f"GPU {i + 1}\n{lab}: 5.6/s", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C[ck])
ax.add_patch(Rectangle((2.6, 0.8), 1.2, 0.7, facecolor=C["memory_l"],
                       edgecolor=C["memory"], lw=1.8))
ax.text(3.2, 1.15, "GPU 3\ndecode: 10/s", ha="center", va="center", fontsize=10.5,
        fontweight="bold", color=C["memory"])
for yy in (1.65, 0.65):
    ax.add_patch(FancyArrowPatch((1.32, yy), (2.58, 1.15), arrowstyle="-|>",
                                 mutation_scale=12, color=C["network"], lw=1.6))
ax.text(1.95, 1.85, "KV cache", ha="center", fontsize=10, color=C["network"],
        fontweight="bold")
ax.text(2.0, -0.05, "10 requests/s on 3 GPUs: 3.3 each, 2.1× colocated", ha="center",
        fontsize=10.5, fontweight="bold", color=C["data"])
ax.set_xlim(0, 4.0)
ax.set_ylim(-0.3, 2.1)
hide_axis(ax)
sp.save(fig, out / "distserve_pool.svg")

sp.hbars(out / "split_link.svg", [
    ("Cache traffic at 10 req/s", 11.3, "network", "11.3 GB/s"),
    ("NVLink, A100", 600, "memory", "600 GB/s"),
], "", log=False, size=(5.0, 1.5),
    xlabel="GB/s (OPT-66B, 512-token prompts; DistServe)")


# ------------------------------------------- API prices (book figure 9 variant)
# Slide variant of MS figure 9, drawn from the same series the book uses:
# mlsysim/literature/data/series/llm_api_input_prices.csv (list price per
# million input tokens, with dated vendor sources). The case study's
# derived floor, $0.066 per million tokens (MS Sec. 13.10.3), is drawn on it.
import csv  # noqa: E402
from datetime import date  # noqa: E402

PRICES = Path(__file__).resolve().parents[4] / "mlsysim" / "mlsysim" / "literature" / "data" / "series" / "llm_api_input_prices.csv"
rows = list(csv.DictReader(PRICES.open()))
OFFSET = {"Legacy GPT-3.5": (4, -13, "left"), "GPT-3.5 Turbo": (6, -12, "left"),
          "GPT-4 (Original)": (6, 3, "left"), "Claude 3 Opus": (6, 4, "left"),
          "Claude 3 Haiku": (-6, -12, "right"), "GPT-4o": (6, 2, "left"),
          "GPT-4o-mini": (6, 3, "left"), "Gemini 1.5 Flash": (-6, 4, "right"),
          "DeepSeek-V3": (-6, 6, "right")}
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for r in rows:
        d = date.fromisoformat(r["date"])
        x = d.year + (d.timetuple().tm_yday - 1) / 365.0
        y = float(r["usd_per_mtok_input"])
        ax.plot([x], [y], "o", ms=7, color=C["memory"], mec="white", zorder=5)
        dx, dy, ha = OFFSET[r["model"]]
        ax.annotate(r["model"], (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=9, color=C["text"])
    if k >= 2:
        ax.axhline(0.066, color=C["data"], lw=2.5)
        ax.text(2022.75, 0.058, "\\$0.066: case study's floor (derived)", ha="left",
                va="top", fontsize=9.5, color=C["data"], fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.02, 80)
    ax.set_xlim(2022.7, 2025.4)
    ax.set_xticks([2023, 2024, 2025])
    ax.set_xticklabels(["2023", "2024", "2025"])
    ax.set_yticks([0.1, 1, 10])
    ax.set_yticklabels(["\\$0.10", "\\$1", "\\$10"])
    ax.set_ylabel("List price, \\$ per million input tokens", fontsize=10)
    tag(ax, "After Model Serving, figure 9 (vendor list prices, historical)", y=-0.12)
    sp.save(fig, out / f"api_price_step{k}.svg")

# ------------------------------------------------------ joules per token
sp.hbars(out / "joules_token.svg", [
    ("Batch 1", 0.99, "danger", "0.99 J per token"),
    ("Batch 32", 0.12, "data", "0.12 J per token"),
], "", log=False, value_fmt="{:.2f}", size=sp.SHORT,
    xlabel="GPT-2 XL decode, joules per token (derived: 1 kW node, constant power)")

print("figures written")
