#!/usr/bin/env python3
"""Figures for Volume II, Lecture 10 (Serving at Scale).

Every number on a figure comes from MLSysIM through calcs.py (val/num) or from
the MLSysIM scenario that owns it; this file draws. Values computed by a
scenario are labeled "derived" on the figure or on its slide; scenario inputs
are labeled "illustrative". Layout coordinates are the only literals here.

Sources (all via MLSysIM)
  - Hook: Scenarios.LlmServingFirstYear (Napkin Math 10.1).
  - Serving tax: Scenarios.InteractiveRequestServingTax (Sec. 10.3.2).
  - InferenceX (SemiAnalysis, Apache-2.0), DeepSeek-R1-0528, FP8, snapshot
    2026-10-12: Scenarios.InferenceXServingFrontier over
    vol2/10_inference/data/inferencex_dsr1_fp8_1k1k.csv. Frontier = configurations no
    other configuration on the same accelerator beats on both axes.
  - Knee: Scenarios.VisionBatchingKnee, Scenarios.LlmDecodeBatchingKnee.
  - Four-request batch: Scenarios.FourRequestDecodeBatch (Napkin Math 10.3).
  - Azure LLM inference traces (CC BY): Scenarios.LlmChatTrace2023,
    LlmCodeTrace2023 (Patel et al., ISCA 2024) and LlmChatWeekTraffic2024
    (Stojkovic et al., HPCA 2025), series in Literature.
  - KV admission: Scenarios.Llama3_70BLongContextAdmissionDgxH100 and
    Llama3_70BChatAdmissionDgxH100 (Napkin Math 10.5).
  - Precision: Scenarios.Llama2_70BPrecisionDividendGqaTp8 (Napkin Math 10.7).
  - Roofline: Hardware.Cloud.H100 ridge point; decode intensity from
    Scenarios.Llama3_70B_DecodeH100.
  - Chunked prefill: Scenarios.LongPromptChunkedPrefill (Example 10.5).
  - Tensor parallelism: Scenarios.Llama2_70BPrefillTensorParallel8 (Example 10.6).
  - Decode budget and speculation: Scenarios.Llama3_70B_ReplicaDecodeBudget,
    Llama3_70B_ReplicaSpeculation (Sec. 10.15.1, table 26).
  - Two choices: Scenarios.RandomVsTwoChoicePlacement (simulated, seed 7).
  - Fan-out: Scenarios.ServingFanout100Leaves (Sec. 10.11.1).
  - Cost per token: Scenarios.H100OutputTokenCost (hypothetical price).

Run from this folder: python3 make_figs.py   (then make deck V=2 D=10_inference)
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcs as K  # noqa: E402  (records every value; figures read them)
from deckvals import val, num  # noqa: E402
from mlsysim.physics.reliability import calc_all_up_availability  # noqa: E402

C = sp.C
plt = sp.plt
out = Path(__file__).parent


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
    rows = [("Training", num("hook_train_m1"), "neutral", f"\\${val('hook_train_m1')}M, once"),
            ("Serving,\nfirst year", num("hook_serve_m"), "compute",
             f"\\${val('hook_serve_m')}M  ({val('hook_ratio')}×)")]
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
    model, outside, target = num("tax_model_ms"), num("tax_outside_ms"), num("tax_target_ms")
    ax.barh(0, model, color=C["compute"], height=0.6)
    ax.text(model / 2, 0, f"model execution  {val('tax_model_ms')} ms", ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")
    ax.barh(0, outside, left=model, color=C["neutral_l"], height=0.6)
    ax.text(model + outside / 2, 0, f"{val('tax_outside_ms')} ms", ha="center", va="center", fontsize=12,
            fontweight="bold")
    ax.axvline(target, color=C["danger"], ls="--", lw=2)
    ax.text(target, 0.42, f"{val('tax_target_ms')} ms target ", color=C["danger"], fontsize=11,
            fontweight="bold", va="bottom", ha="right")
    if k >= 2:
        ax.barh(0, num("tax_exchanges_high_ms"), left=model, color=C["network"], height=0.6)
        ax.annotate(f"{val('tax_exchanges')} exchanges: {val('tax_exchanges_low_ms')} to "
                    f"{val('tax_exchanges_high_ms')} ms", (model + 1, -0.3), xytext=(40, -0.95),
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
IX = K.IX


HW = [("h100", "H100", "neutral"), ("h200", "H200", "memory"), ("mi355x", "MI355X", "data"),
      ("b200", "B200", "compute"), ("gb200", "GB200 NVL72", "network")]
B200 = IX.solve("b200", num("tps_high"))
b200_peak = (B200.peak_interactivity, B200.peak_throughput)
b200_100 = B200.best_at


LABX = {"h100": 75, "h200": 150, "mi355x": 185, "b200": 225, "gb200": 95}
LABDY = {"h100": 0.4, "h200": 0.4, "mi355x": 0.4, "b200": 2.0, "gb200": 2.0}


def ixplot(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    shown = ["b200"] if k == 1 else [h for h, _, _ in HW]
    for h, lab, ck in HW:
        rows = IX.rows(h)
        f = IX.frontier(h)
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
        ax.annotate(f"{val('b200_peak')} at {val('b200_peak_user')} tok/s per user",
                    b200_peak, xytext=(25, 2), textcoords="offset points", fontsize=11,
                    fontweight="bold", color=C["compute"])
        x100 = [r for r in IX.rows("b200") if r["output_tok_s_gpu"] == b200_100][0]["interactivity_tok_s_user"]
        ax.plot(x100, b200_100, "o", ms=9, color=C["compute"], mec="white")
        ax.annotate(f"{val('b200_at_high')} at {val('tps_high')}+: {val('b200_span')}× less", (x100, b200_100),
                    xytext=(-20, -38),
                    textcoords="offset points", fontsize=11, fontweight="bold",
                    color=C["danger"])
    if k >= 3:
        t50 = num("tps_mid")
        ax.axvline(t50, color=C["text"], ls="--", lw=1.2)
        for key, ck in (("h100_at_mid", "neutral"), ("gb200_at_mid", "network")):
            v = IX.best_at(key.split("_")[0], t50)
            ax.plot(t50, v, "o", ms=8, color=C[ck], mec="white", zorder=5)
            ax.text(t50 - 4, v, val(key), ha="right", va="center", fontsize=11, fontweight="bold",
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
    for lab, kn, ck, xy in ((f"vision: knee at\n{val('knee_vis_t0')} ms / {val('knee_vis_treq')} ms = "
                             f"{val('knee_vis')}", num("knee_vis"), "data", (2.6, 0.2)),
                            (f"LLM decode: knee at\n{val('knee_llm_t0')} ms / {val('knee_llm_treq')} ms = "
                             f"{val('knee_llm')}", num("knee_llm"), "memory", (120, 0.2))):
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
    rows = IX.sweep("b200", framework="sglang")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.8, 2.7), sharex=True)
    xs = [r["conc"] for r in rows]
    a1.scatter(xs, [r["output_tok_s_gpu"] for r in rows], s=28, color=C["compute"], zorder=3)
    a2.scatter(xs, [r["interactivity_tok_s_user"] for r in rows], s=28, color=C["memory"], zorder=3)
    a1.set_ylabel("tok/s per GPU", fontsize=11)
    a2.set_ylabel("tok/s per user", fontsize=11)
    a1.text(1, 900, f"{val('conc_lo_gpu')} to {val('conc_hi_gpu')} per GPU ({val('conc_gpu_gain')}×)",
            fontsize=11, fontweight="bold", color=C["compute"])
    a2.text(40, 150, f"{val('conc_lo_user')} to {val('conc_hi_user0')} per user ({val('conc_user_drop')}× slower)",
            fontsize=11, fontweight="bold", color=C["memory"])
    a1.set_ylim(0, 1300)
    a2.set_ylim(0, 260)
    a2.set_xscale("log", base=2)
    a2.set_xticks([1, 4, 16, 64, 256, 512])
    a2.set_xticklabels(["1", "4", "16", "64", "256", "512"])
    a2.set_xlabel("Concurrent requests (8 B200 GPUs, SGLang, colocated)")
    sp.save(fig, out / "conc_measured.svg")


conc()

# =================================== Azure trace: output lengths and slot waste
gv = np.array(K.CHAT.output_tokens)


def lengths():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.4), gridspec_kw={"width_ratios": [1.5, 1]})
    a1.hist(gv, bins=np.arange(0, num("chat_max_output") + 1, 25), color=C["memory"])
    for v, lab, dy in ((K.chat.median_output, f"median {val('chat_med_output')}", 0.92),
                       (K.chat.mean_output, f"mean {val('chat_mean_output')}", 0.70)):
        a1.axvline(v, color=C["text"], lw=1.2, ls="--")
        a1.text(v + 15, a1.get_ylim()[1] * dy, lab, fontsize=10.5, fontweight="bold")
    a1.set_xlabel("Output tokens per chat request")
    a1.set_ylabel("Requests")
    a1.set_xlim(0, num("chat_max_output"))
    bars = [("Ch. example", "cb_idle_pct", K.BATCH.idle_share, "neutral"),
            ("Chat trace", "chat_idle_pct", K.chat.fixed_batch_idle_share, "memory"),
            ("Code trace", "code_idle_pct", K.code.fixed_batch_idle_share, "compute")]
    for i, (lab, key, share, ck) in enumerate(bars):
        v = 100 * share
        a2.barh(2 - i, v, color=C[ck], height=0.6)
        a2.text(v + 2, 2 - i, f"{val(key)}%", va="center", fontsize=12, fontweight="bold", color=C[ck])
        a2.text(-3, 2 - i, lab, ha="right", va="center", fontsize=11, fontweight="bold")
    a2.set_xlim(0, 90)
    a2.set_yticks([])
    a2.spines["left"].set_visible(False)
    a2.set_xlabel(f"Slot time idle, batch of {val('fixed_batch')}")
    sp.save(fig, out / "lengths.svg")


lengths()

# ===================================== continuous batching schedule (Napkin 10.3)
OUTS = K.BATCH.output_lengths
LONGEST = K.BATCH.max_output


def cb(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    names = ["R1", "R2", "R3", "R4"]
    nxt = {0: "R5", 2: "R6", 3: "R7"}
    for s, L in enumerate(OUTS):
        y = 3 - s
        ax.add_patch(Rectangle((0, y), L, 0.7, color=C["memory"]))
        ax.text(L / 2, y + 0.35, f"{names[s]}  {L}", ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")
        if L < LONGEST:
            if k == 1:
                ax.add_patch(Rectangle((L, y), LONGEST - L, 0.7, facecolor="white",
                                       edgecolor=C["danger"], hatch="///", lw=1.0))
            else:
                ax.add_patch(Rectangle((L, y), LONGEST - L, 0.7, color=C["data"]))
                ax.text(L + (LONGEST - L) / 2, y + 0.35, nxt[s] if L > 120 else f"{nxt[s]} joins at {L}", ha="center",
                        va="center", color="white", fontsize=11, fontweight="bold")
    if k == 1:
        ax.text(205, 1.9, f"idle: {val('cb_idle_iters')} of {val('cb_slot_iters')}\nslot-iterations\n= "
                f"{val('cb_idle_pct')}%", fontsize=11,
                color=C["danger"], fontweight="bold", va="center")
    else:
        ax.text(205, 1.9, f"{val('cb_advanced')} requests\nadvance in the\nsame {val('cb_max')}\niterations",
                fontsize=11,
                color=C["data"], fontweight="bold", va="center")
    ax.set_xlim(0, 260)
    ax.set_ylim(-0.1, 3.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlabel(f"Decode iteration ({val('cb_iter_ms')} ms each, illustrative)")
    sp.save(fig, out / f"cb_step{k}.svg")


for k in (1, 2):
    cb(k)

# ============================================ KV admission: 12 versus 1,397
FREE, LONG, N_LONG = num("kv_budget_gb0"), num("kv_long_gb"), int(num("kv_long_n"))


def kv(k):
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    W = 1.5
    ax.add_patch(Rectangle((0, 0), W, FREE, fill=False, ec=C["text"], lw=1.2))
    for i in range(N_LONG):
        ax.add_patch(Rectangle((0.06, i * LONG + 1.5), W - 0.12, LONG - 2.4, color=C["memory"]))
    ax.text(W / 2, 260, f"{val('kv_long_n')} requests\n× {val('kv_long_gb')} GB\n(128K tokens)", ha="center",
            va="center",
            color="white", fontsize=12, fontweight="bold",
            bbox=dict(fc=C["memory"], ec="none", pad=2))
    ax.plot([0, 2 * W + 0.6], [FREE, FREE], color=C["danger"], lw=2)
    ax.text(W + 0.3, 545, f"{val('kv_budget_gb0')} GB free for KV, {val('node_gpus')}× H100", ha="center",
            fontsize=11.5,
            color=C["danger"], fontweight="bold")
    a = 1.0 if k >= 2 else 0.0
    x2 = W + 0.6
    ax.add_patch(Rectangle((x2, 0), W, FREE, fill=False, ec=C["text"], lw=1.2, alpha=a))
    ax.add_patch(Rectangle((x2 + 0.06, 1.5), W - 0.12, FREE - 3, color=C["data"], alpha=a))
    ax.text(x2 + W / 2, 260, f"{val('kv_chat_n')} requests\n× {val('kv_chat_mb')} MB\n(median chat,\n"
            f"{val('chat_tokens')} tokens)",
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
    ax.text(16.2, 1.35, f"{len(used)} of 16 slots\nhold tokens", va="center", fontsize=10.5,
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
    rows = [("70B FP16", "w70_fp16_gb", "memory"), ("70B INT4", "w70_int4_gb", "data"),
            ("175B INT4", "w175_int4_gb", "compute")]
    gpu = num("prec_hbm_gb")
    for i, (lab, key, ck) in enumerate(rows):
        v = num(key)
        a1.barh(2 - i, v, color=C[ck], height=0.6)
        a1.text(max(v + 3, gpu + 4), 2 - i, f"{val(key)} GB", va="center", fontsize=11.5, fontweight="bold",
                color=C[ck])
        a1.text(-4, 2 - i, lab, ha="right", va="center", fontsize=11, fontweight="bold")
    a1.axvline(gpu, color=C["danger"], ls="--", lw=1.8)
    a1.text(gpu + 2, 2.45, f"one {val('prec_hbm_gb')} GB GPU", color=C["danger"], fontsize=10, fontweight="bold")
    a1.set_xlim(0, 230)
    a1.set_ylim(-0.6, 2.8)
    a1.set_yticks([])
    a1.spines["left"].set_visible(False)
    a1.set_xlabel("Weight bytes")
    for i, (lab, key, ck) in enumerate((("FP16", "prec_batch", "neutral"), ("INT4 W,\nINT8 KV", "prec_batch_low", "data"))):
        v = num(key)
        a2.barh(1 - i, v, color=C[ck], height=0.6)
        a2.text(v + 15, 1 - i, val(key), va="center", fontsize=12, fontweight="bold", color=C[ck])
        a2.text(-15, 1 - i, lab, ha="right", va="center", fontsize=11, fontweight="bold")
    a2.text(380, -0.75, f"{val('prec_dividend')}×", fontsize=13, fontweight="bold", color=C["data"])
    a2.set_xlim(0, 1150)
    a2.set_ylim(-1.0, 1.6)
    a2.set_yticks([])
    a2.spines["left"].set_visible(False)
    a2.set_xlabel(f"Max batch, 70B GQA, {val('prec_tp')}× H100")
    sp.save(fig, out / "precision.svg")


precision()

# =============================================================== roofline
def roofline(k):
    fig, ax = plt.subplots(figsize=(4.3, 2.7))
    R = num("ridge")
    xs = np.logspace(-0.3, 4, 300)
    roof = np.minimum(1.0, xs / R)
    ax.plot(xs[xs <= R], roof[xs <= R], color=C["memory"], lw=3)
    ax.plot(xs[xs >= R], roof[xs >= R], color=C["compute"], lw=3)
    ax.fill_between(xs[xs <= R], 1e-3, roof[xs <= R], color=C["memory_l"], alpha=0.6)
    ax.fill_between(xs[xs >= R], 1e-3, roof[xs >= R], color=C["compute_l"], alpha=0.6)
    ax.axvline(R, color=C["neutral"], lw=1, ls="--")
    ax.text(R * 1.1, 0.0045, f"ridge {val('ridge')}\n(H100 FP16,\nderived)", color=C["neutral"], fontsize=10)
    pts = [(f"decode, B = {val('decode_intensity')}", num("decode_intensity"), "memory", (10, -16), "left"),
           (f"decode, B = {val('knee_llm')}", num("knee_llm"), "memory", (-10, 8), "right"),
           (f"prefill,\n{val('ix_isl')}-token\nprompt", num("ix_isl"), "compute", None, "left")]
    for i, (lab, x, ck, off, ha) in enumerate(pts[:k]):
        y = min(1.0, x / R)
        ax.plot([x], [y], "o", ms=11, color=C[ck], mec="white", mew=1.5, zorder=5)
        if off is None:  # label in open space right of the ridge, with a leader line
            ax.annotate(lab, (x, y), xytext=(R * 1.35, 0.2), textcoords="data",
                        ha=ha, va="center", fontsize=11.5, fontweight="bold",
                        color=C[ck], linespacing=1.0,
                        arrowprops=dict(arrowstyle="-", color=C[ck], lw=1.2,
                                        shrinkA=2, shrinkB=6))
            continue
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", ha=ha,
                    fontsize=11.5, fontweight="bold", color=C[ck])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 2e4)
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
    ax.text(6, 1.6, f"{val('chunk_prompt')}-token prefill", ha="center", va="center", color="white",
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
        row(0.0, f"{val('chunk_size')}-token\nchunks", segs)
        ax.text(8.5, -0.35, f"each iteration: decode tokens + one chunk  (≥ {val('chunk_iters')} iterations)",
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
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.9), sharey=True)
    for ax, (h, lab) in zip(axs, (("b200", "B200"), ("h200", "H200"))):
        for j, t in enumerate(IX.targets):
            j = 1.4 * j
            for d, ck, dx in ((1.0, "network", -0.22), (0.0, "neutral", 0.22)):
                v = IX.best_at(h, t, disagg=d)
                ax.bar(j + dx, v, width=0.38, color=C[ck])
                # split-pool label hangs left of the pair, colocated right,
                # so the two values never collide at similar heights
                ax.text(j + dx + (0.19 if dx < 0 else -0.19), v * 1.12, f"{v:,.0f}",
                        ha="right" if dx < 0 else "left", va="bottom", fontsize=11,
                        fontweight="bold", color=C[ck])
        ax.set_yscale("log")
        ax.set_ylim(20, 30000)
        ax.set_xticks([0, 1.4, 2.8])
        ax.set_xticklabels([val("disagg_t1"), val("disagg_t2"), val("disagg_t3")])
        ax.set_xlabel("tok/s per user, at least")
        ax.set_title(lab, fontsize=13, fontweight="bold", loc="left")
        ax.set_xlim(-0.95, 3.55)
        ax.minorticks_off()
    axs[0].set_ylabel("top tok/s per GPU")
    axs[1].text(0.55, 0.93, "split pools", transform=axs[1].transAxes, color=C["network"],
                fontsize=11, fontweight="bold")
    axs[1].text(0.55, 0.80, "colocated", transform=axs[1].transAxes, color=C["neutral"],
                fontsize=11, fontweight="bold")
    sp.save(fig, out / "disagg.svg")


disagg()

# ================================================ tensor parallel per layer
def tp(k):
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    at, ff, ar = num("tp_attn_ms"), num("tp_ffn_ms"), num("tp_ar_ms")
    at8, ff8 = num("tp_attn_shard_ms"), num("tp_ffn_shard_ms")
    ax.barh(1, at, color=C["compute"], height=0.55)
    ax.barh(1, ff, left=at, color=C["compute_l"], height=0.55, ec=C["compute"])
    ax.text(at / 2, 1, f"attn {val('tp_attn_ms')}", ha="center", va="center", color="white", fontsize=11,
            fontweight="bold")
    ax.text(at + ff / 2, 1, f"FFN {val('tp_ffn_ms')}", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(at + ff + 0.5, 1, f"{val('tp_layer_ms')} ms", va="center", fontsize=12, fontweight="bold")
    ax.text(-0.5, 1, "1 GPU", ha="right", va="center", fontsize=12, fontweight="bold")
    if k >= 2:
        ax.barh(0, at8, color=C["compute"], height=0.55)
        ax.barh(0, ar, left=at8, color=C["network"], height=0.55)
        ax.barh(0, ff8, left=at8 + ar, color=C["compute_l"], height=0.55, ec=C["compute"])
        ax.barh(0, ar, left=at8 + ar + ff8, color=C["network"], height=0.55)
        ax.text(4.8, 0, f"{val('tp_layer_par_ms')} ms = {val('tp_compute_ms')} compute + 2 × {val('tp_ar_ms')} "
                "AllReduce", va="center", fontsize=11.5, fontweight="bold", color=C["network"])
        ax.text(-0.5, 0, f"{val('tp_size')}-way TP", ha="right", va="center", fontsize=12, fontweight="bold")
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
COLS = [("Baseline", K.BUDGET.baseline, "budget_base1"),
        ("INT4 weights", K.BUDGET.after_precision, "budget_prec"),
        ("+ fusion, replay", K.BUDGET.after_fusion, "budget_fused")]
TARGET = num("tpot_ms")


def budget(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    n = min(k, 3)
    for i, (lab, vals, key) in enumerate(COLS):
        y = 2 - i
        left = 0
        a = 1.0 if i < n else 0.0
        for (pname, ck), v in zip(PARTS, vals):
            ax.barh(y, v, left=left, color=C[ck], height=0.6, alpha=a, ec="white", lw=0.6)

            left += v
        ax.text(-1, y, lab, ha="right", va="center", fontsize=11.5, fontweight="bold", alpha=a)
        ax.text(left + 1, y, f"{val(key)} ms", va="center", fontsize=12, fontweight="bold",
                alpha=a, color=C["danger"] if num(key) > TARGET else C["data"])
    for (pname, ck), xk in zip(PARTS, (0, 14, 28.5, 57, 79)):
        ax.text(xk, 2.62, pname, fontsize=10, color=C[ck], fontweight="bold")
    ax.axvline(TARGET, color=C["danger"], ls="--", lw=1.8)
    ax.text(TARGET + 0.8, -2.3, f"{val('tpot_ms')} ms target", color=C["danger"], fontsize=10.5, fontweight="bold")
    if k >= 4:
        ax.barh(-1, num("spec_per_token"), color=C["data"], height=0.6)
        ax.text(52, -0.85, f"{val('spec_per_token')} ms per token on average,", va="center", fontsize=11,
                fontweight="bold", color=C["data"])
        ax.text(52, -1.4, f"but blocks arrive {val('spec_round')} ms apart", va="center", fontsize=11,
                fontweight="bold", color=C["danger"])
        ax.text(-1, -1, "Speculate", ha="right", va="center", fontsize=11.5, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-2.6, 3.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"ms per batch-one decode step, 70B on {val('prec_tp')}× H100 (constructed)")
    sp.save(fig, out / f"budget_step{k}.svg")


for k in (1, 2, 3, 4):
    budget(k)

# ======================================================= power of two choices
P2C = K.p2c
NS, R1, R2 = P2C.bins, P2C.max_load[1], P2C.max_load[2]


def p2c():
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    ax.plot(NS, R1, "o-", color=C["neutral"], lw=2.6, ms=7)
    ax.plot(NS, R2, "o-", color=C["memory"], lw=2.6, ms=7)
    ax.text(NS[-1], R1[-1] + 0.6, f"random: {val('p2c_random')}", ha="right", va="bottom", fontsize=11.5,
            fontweight="bold", color=C["neutral"])
    ax.text(NS[-1], R2[-1] - 1.0, f"two probes: {val('p2c_two')}", ha="right", va="top", fontsize=11.5,
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
    F = K.FANOUT
    leaves = F.n_components
    N = np.logspace(0, 3, 200)
    q1 = 1 - F.availability                      # each leaf slow at its own p99
    tiers = ((q1, "leaf slow {}% (p{})", "danger", (85, 4, "right")),
             (q1 / 10, "{}% (p{})", "compute", (118, 8, "left")),
             (q1 / 100, "{}% (p{})", "data", (118, -9, "left")))
    for q, lab, ck, (dx, dy, ha) in tiers:
        p = [1 - calc_all_up_availability(1 - q, n) for n in N]   # the scenario's law, along N
        ax.plot(N, 100 * np.array(p), color=C[ck], lw=2.6)
        v = 100 * (1 - F.variant(availability=1 - q).all_up)
        ax.plot([leaves], [v], "o", ms=8, color=C[ck], mec="white", zorder=5)
        ax.text(1150, 100 * p[-1], lab.format(f"{100 * q:g}", f"{100 - 100 * q:g}"), fontsize=10.5,
                color=C[ck], fontweight="bold", va="center")
        ax.text(dx, v + dy, f"{v:.0f}%" if v >= 5 else f"{v:.1f}%", fontsize=11,
                color=C[ck], fontweight="bold", ha=ha)
    ax.axvline(leaves, color=C["neutral"], ls="--", lw=1)
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
    W = K.WEEK
    r = W.series.entries
    t = [datetime.strptime(x[W.bin_column], "%Y-%m-%d %H:%M") for x in r]
    v = np.array([x[W.count_column] for x in r]) / W.bin_width.to("second").magnitude
    hours = np.array([(x - t[0]).total_seconds() / 3600 for x in t])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.2, 3.3), gridspec_kw={"height_ratios": [1.2, 1]})
    a1.plot(hours / 24, v, color=C["memory"], lw=1.6)
    a1.set_ylabel("req/s", fontsize=11)
    a1.set_xlim(0, 7)
    a1.set_ylim(0, 95)
    a1.set_xticks(np.arange(0, 7) + 0.5)
    a1.set_xticklabels(["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
    a1.text(1.6, 84, f"weekday peak {val('week_peak')} req/s, about {val('week_swing0')}× the night trough",
            fontsize=10.5,
            fontweight="bold", color=C["memory"])
    tc = [datetime.strptime(x[K.CODE.timestamp_column][:26], "%Y-%m-%d %H:%M:%S.%f")
          for x in K.CODE.series.entries]
    secs = np.array([(x - tc[0]).total_seconds() for x in tc])
    per = np.bincount(secs.astype(int))
    a2.plot(np.arange(len(per)) / 60, per, color=C["compute"], lw=0.8)
    a2.set_xlabel("Minutes (one hour, code trace); top: one week, chat trace, 10-min means")
    a2.set_ylabel("req per s", fontsize=11)
    a2.text(16, per.max() * 0.92, f"mean {val('code_per_s')}/s, busiest second {val('code_peak_s')}", fontsize=10.5,
            fontweight="bold", color=C["compute"])
    a2.set_xlim(0, 58)
    a2.set_ylim(0, per.max() * 1.18)
    sp.save(fig, out / "traffic.svg")


traffic()

# ================================================ tenant quota (Example 10.9)
sp.hbars(out / "quota.svg",
         [("no quota", num("nn_p99_hi"), "danger",
           f"{val('nn_p99_lo')} to {val('nn_p99_hi')} ms"),
          (f"quota at {val('nn_quota')} QPS", num("nn_protected_ms"), "data",
           f"{val('nn_protected_ms')} ms")],
         "ms", log=False, size=(3.0, 1.5), fontsize=15,
         xlabel=f"other tenants' p99 (ms); SLO {val('nn_slo_ms')} ms")

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
def synth():
    rows = [(f"{val(f'tps_{lvl}')} tok/s\nper user", K.COST.cost_per_million(K.COST.targets[i - 1]).magnitude, ck,
             f"\\${val(f'syn_cost{i}')}  ({val(f'syn_tok{i}')} tok/s/GPU)")
            for i, lvl, ck in ((1, "low", "data"), (2, "mid", "compute"), (3, "high", "danger"))]
    sp.hbars(out / "synth_cost.svg", rows, "", log=True, size=(5.8, 1.9),
             xlabel="\\$ per million output tokens, H100 (derived)")


synth()
