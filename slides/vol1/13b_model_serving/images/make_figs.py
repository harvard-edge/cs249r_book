#!/usr/bin/env python3
"""Figures for Lecture 13b (Model Serving, Part 2: Serving a Stream of Tokens).

Every number comes from calcs.py (imported as K), which reads MLSysIM:
the token-serving Scenarios.* and the cited results in Literature (MLPerf
Inference v5.0 Llama 2 70B; Splitwise; DistServe; PagedAttention; the API
price series behind the chapter's figure 9). Labels print the formatted
strings calcs.py records (deckvals.val), so a figure and its slide show the
same digits. Values computed here are labeled "derived" on the figure or its
slide. Illustrative geometry (the fragmentation strip, the paging example,
slot placement in the schedules) is layout and stays in this file.

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

import contextlib  # noqa: E402
import io  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    import calcs as K  # noqa: E402  every derived value comes from calcs.py
from deckvals import val, num  # noqa: E402

C = sp.C
plt = sp.plt
out = Path(__file__).parent


def cm(w, h):
    """Figure size for a slot of w by h centimeters on the 16 x 9 cm page."""
    return (w / 2.54, h / 2.54)


def save_slot(fig, path):
    """Save at exactly the figure size, so the slide places it at scale 1."""
    fig.savefig(path, format="svg", facecolor="white")
    plt.close(fig)


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
HOOK = [("No deadline\n(Offline)", K.MLP_OFFLINE, "neutral", val("llama_off")),
        (f"TTFT {val('llama_conv_ttft_s')} s,\nTPOT {val('llama_conv_tpot')} ms", K.MLP_SERVER, "compute",
         f"{val('llama_conv')}  ({val('llama_conv_pct')}%)"),
        (f"TTFT {val('llama_int_ttft')} ms,\nTPOT {val('llama_int_tpot')} ms", K.MLP_INTER, "danger",
         f"{val('llama_int')}  ({val('llama_int_pct')}%)")]


def hook_fig(path, k, size=(5.8, 2.6), note=True, fs=12, xlabel=True, rows=None):
    fig, ax = plt.subplots(figsize=size)
    for i, (lab, v, ck, txt) in enumerate(rows or HOOK):
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
        ax.text(39, -0.75, f"The same server keeps {val('hook_fewer')} times fewer tokens "
                f"per second when TPOT tightens {val('tpot_tighten')} times.",
                ha="right", va="center", fontsize=11, color=C["danger"],
                fontweight="bold", alpha=1.0 if k >= 4 else 0.0)
    ax.set_yticks([])
    ax.set_xlim(0, 40)
    ax.set_ylim(-1.1 if note else -0.5, 2.5)
    ax.spines["left"].set_visible(False)
    if xlabel:
        ax.set_xlabel(f"Thousand tokens/s, Llama 2 70B on one {val('mlperf_gpus')}× H100 server")
    else:
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    sp.save(fig, path)


for k in (1, 2, 3, 4):
    hook_fig(out / f"hook_step{k}.svg", k)
# Synthesis thumbnail: three values in large type, no axis title.
THUMB = [("Offline", K.MLP_OFFLINE, "neutral", val("llama_off")),
         (f"{val('llama_conv_tpot')} ms", K.MLP_SERVER, "compute", val("llama_conv")),
         (f"{val('llama_int_tpot')} ms", K.MLP_INTER, "danger", val("llama_int"))]
hook_fig(out / "hook_thumb.svg", 4, size=(3.6, 1.6), note=False, fs=16, xlabel=False,
         rows=THUMB)

# ------------------------------------------- recall thumbnail (Part 1 hook)
sp.hbars(out / "recall_cost.svg", [
    ("Batch 1", num("recall_b1_usd"), "danger", f"\\${val('recall_b1_usd')}"),
    (f"Batch {val('batch_8')}", num("recall_b8_usd"), "compute", f"\\${val('recall_b8_usd')}"),
], "", log=False, size=(4.0, 1.4),
    xlabel="ResNet-50, \\$ per million requests (modeled)")

# ---------------------------------------------------------- TTFT and TPOT
for k in (1, 2, 3, 4):
    fig, ax = plt.subplots(figsize=(5.9, 2.6))
    rows = [("DGX-H100", 1.3, K.H_TTFT, K.H_TBT, K.H_E2E),
            ("DGX-A100", 0.0, K.A_TTFT, K.A_TBT, K.A_E2E)]
    for name, y, ttft, tbt, e2e in rows:
        a = 1.0 if (y > 0.5 or k >= 3) else 0.0
        ax.barh(y, ttft, color=C["compute"], height=0.42, alpha=a)
        for t in np.arange(ttft + tbt, e2e, tbt):
            ax.plot([t, t], [y - 0.2, y + 0.2], color=C["memory"], lw=1.1, alpha=a)
        ax.text(-60, y, name, ha="right", va="center", fontsize=12,
                fontweight="bold", alpha=a)
        ax.text(e2e + 50, y, f"{e2e:,.0f} ms", va="center", fontsize=11,
                fontweight="bold", alpha=a)
        ax.text(600, y + 0.3, f"first token {ttft:.0f} ms, then one every {tbt:.0f} ms",
                fontsize=10.5, color=C["memory"], fontweight="bold", alpha=a)
    if k >= 2:
        ax.text(600, 0.82, f"decode: {val('sw_decode_pct')}% of the request (derived)", ha="left",
                va="center", fontsize=10.5, color=C["memory"], fontweight="bold")
    if k >= 4:
        ax.axvline(num("target_ttft"), color=C["danger"], lw=2, ls="--")
        ax.text(600, -0.62, f"book target: TTFT < {val('target_ttft')} ms;  TPOT < {val('target_tpot')} ms "
                f"({val('target_rate')} tokens/s)",
                fontsize=10.5, color=C["danger"], fontweight="bold", va="center")
    ax.set_xlim(0, 5600)
    ax.set_ylim(-0.85, 1.85)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Milliseconds, one chat request (Llama 2 70B, median, no batching)")
    sp.save(fig, out / f"ttft_step{k}.svg")

# ------------------------------------------------- prefill against decode
# Slot: 10.4 x 5.0 cm beside the H100 photo. Values from calcs.py.
PHASE_PRE = [(f"GPT-2 XL,\n{val('gpt2_prompt')} tokens", K.GPT2_PREFILL_MS, val("gpt2_prefill_ms")),
             (f"Llama 3 8B,\n{val('cs_prompt')} tokens", K.PRE_MS, val("cs_ttft"))]
PHASE_DEC = [("GPT-2 XL", K.GPT2_FLOOR_MS, "memory", f"{val('gpt2_floor_ms')} ms"),
             ("Llama 3 8B,\nINT4", K.L3_TPOT_MS, "memory",
              f"{val('cs_floor_ms')} floor, {val('cs_tpot_ms')}"),
             ("GPT-2 XL,\neager", K.EAGER_MS, "neutral", f"{val('eager_ms')} ms")]
FS = 9
for k in (1, 2, 3, 4):
    fig = plt.figure(figsize=cm(10.4, 5.0))
    ax = fig.add_axes([0.21, 0.46, 0.25, 0.40])
    bx = fig.add_axes([0.71, 0.46, 0.27, 0.40])
    cx = fig.add_axes([0.21, 0.10, 0.77, 0.12])
    for i, (lab, v, vtxt) in enumerate(PHASE_PRE):
        y = 1 - i
        ax.barh(y, v, color=C["compute"], height=0.6)
        ax.text(v + 1.5 if v < 30 else v - 1.5, y, f"{vtxt} ms", va="center",
                ha="left" if v < 30 else "right", fontsize=FS, fontweight="bold",
                color=C["compute"] if v < 30 else "white")
        ax.text(-2, y, lab, ha="right", va="center", fontsize=FS - 0.5, fontweight="bold")
    ax.set_xlim(0, 50)
    ax.set_title(f"Prefill: FLOP ÷ ({val('prefill_eff')}% of peak)", fontsize=FS, loc="left",
                 fontweight="bold", color=C["compute"])
    ax.set_xlabel("ms (sets TTFT)", fontsize=FS - 0.5, labelpad=1)
    a2 = 1.0 if k >= 2 else 0.0
    for i, (lab, v, ck, txt) in enumerate(PHASE_DEC):
        y = 2 - i
        a = a2 if i < 2 else (1.0 if k >= 3 else 0.0)
        bx.barh(y, v, color=C[ck], height=0.6, alpha=a)
        bx.text(v + 0.3 if v < 5 else v - 0.3, y, txt, va="center",
                ha="left" if v < 5 else "right", fontsize=FS - 0.5, fontweight="bold",
                color=C[ck] if v < 5 else "white", alpha=a)
        bx.text(-0.4, y, lab, ha="right", va="center", fontsize=FS - 0.5,
                fontweight="bold", alpha=a)
    bx.set_xlim(0, 10)
    bx.set_title(f"Decode: GB ÷ {val('h100_bw')} TB/s", fontsize=FS, loc="right",
                 fontweight="bold", color=C["memory"], alpha=a2)
    bx.set_xlabel("ms (sets TPOT)", fontsize=FS - 0.5, alpha=a2, labelpad=1)
    if k < 2:
        bx.spines["bottom"].set_alpha(0.0)
        bx.tick_params(axis="x", colors="white")
    for a_ in (ax, bx):
        a_.set_yticks([])
        a_.spines["left"].set_visible(False)
        a_.set_ylim(-0.5, 2.5 if a_ is bx else 1.5)
        a_.tick_params(axis="x", labelsize=8, pad=1)
    a4 = 1.0 if k >= 4 else 0.0
    tp = K.L3_TPOT_MS
    cx.barh(0, tp, color=C["memory"], height=0.8, alpha=a4)
    cx.barh(0, K.TPOT_TARGET_MS - tp, left=tp, color=C["data_l"], edgecolor=C["data"],
            hatch="//", height=0.8, lw=0, alpha=a4)
    cx.text(10.8, 0, f"{val('cs_room_ms')} ms: room to share this read (derived)", ha="center",
            va="center", fontsize=FS - 0.5, fontweight="bold", color=C["data"], alpha=a4,
            bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=a4))
    cx.text(-0.3, 0, f"Llama 3 8B,\n{val('cs_tpot_target')} ms TPOT", ha="right", va="center",
            fontsize=FS - 0.5, fontweight="bold", alpha=a4)
    cx.set_xlim(0, K.TPOT_TARGET_MS)
    cx.set_ylim(-0.6, 0.6)
    cx.set_yticks([])
    cx.set_xticks([0, 5, 10, 15, 20])
    cx.spines["left"].set_visible(False)
    cx.tick_params(axis="x", labelsize=8, pad=1, colors=C["text"] if k >= 4 else "white")
    cx.spines["bottom"].set_alpha(a4)
    save_slot(fig, out / f"phases_step{k}.svg")

# ------------------------------------------------- speculative decoding
# Slot: 14.2 x 2.9 cm. Inputs: MS Sec. 13.9.4 (draft 0.30 ms, verify 0.90 ms).
for k in (1, 2):
    fig = plt.figure(figsize=cm(14.2, 2.9))
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    s = 1.5  # drawing units per ms
    ax.add_patch(Rectangle((0, 1.0), K.DRAFT * s, 0.5, color=C["data"]))
    ax.text(K.DRAFT / 2 * s, 1.25, "draft", ha="center", va="center", fontsize=9,
            color="white", fontweight="bold")
    ax.text(0.0, 1.62, f"Draft proposes {val('spec_k')} tokens ({val('spec_draft_ms')} ms); the target checks all "
            f"{val('spec_k1')} positions in one pass.",
            ha="left", va="bottom", fontsize=9.5, color=C["text"], fontweight="bold")
    ax.add_patch(Rectangle((K.DRAFT * s, 1.0), K.VERIFY * s, 0.5, color=C["memory"]))
    ax.text((K.DRAFT + K.VERIFY / 2) * s, 1.25, f"target: {val('spec_verify_ms')} ms", ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")
    ax.text(K.ROUND * s + 0.05, 1.25, f"{val('spec_round_ms')} ms per round", va="center", fontsize=9.5,
            fontweight="bold")
    a = 1.0 if k >= 2 else 0.0
    marks = [("t1", "data", "kept"), ("t2", "data", "kept"), ("t3", "danger", "rejected"),
             ("t4", "neutral", "dropped"), ("new", "memory", "target's")]
    for j, (lab, ck, word) in enumerate(marks):
        x = j * 0.42
        ax.add_patch(Rectangle((x, 0.30), 0.36, 0.45, facecolor=C[ck + "_l"],
                               edgecolor=C[ck], lw=1.5, alpha=a))
        ax.text(x + 0.18, 0.525, lab, ha="center", va="center", fontsize=9,
                fontweight="bold", color=C[ck], alpha=a)
        ax.text(x + 0.18, 0.10, word, ha="center", va="center", fontsize=8.5,
                color=C[ck], alpha=a)
    ax.text(2.2, 0.525, "one example round: 3 tokens", va="center", fontsize=9.5,
            fontweight="bold", alpha=a)
    ax.set_xlim(0, 3.6)
    ax.set_ylim(-0.05, 1.95)
    hide_axis(ax)
    save_slot(fig, out / f"spec_step{k}.svg")

# ------------------------------------------------------- decode bytes
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    for i, b in enumerate((K.B1, K.B32)):
        cache = b * K.KV_SEQ_GB
        ms = f"{val(f'step_ms_{b}')} ms"
        rate = f"{val(f'rate_{b}')} tokens/s"
        y = 1 - i
        a = 1.0 if (k == 2 or b == 1) else 0.0
        ax.barh(y, K.GPT2_W_GB, color=C["memory"], height=0.55, alpha=a)
        ax.barh(y, cache, left=K.GPT2_W_GB, color=C["data"], height=0.55, alpha=a)
        ax.text(K.GPT2_W_GB + cache + 0.2, y, f"{val(f'step_gb_{b}')} GB: {ms}, {rate}", va="center",
                fontsize=11, fontweight="bold", color=C["text"], alpha=a)
        ax.text(-0.15, y, f"Batch {b}", ha="right", va="center", fontsize=12,
                fontweight="bold", alpha=a)
    ax.text(1.5, 1.45, f"weights {val('gpt2_w_gb')} GB", ha="center", color=C["memory"], fontsize=11,
            fontweight="bold")
    ax.text(8.0, -0.55, f"KV cache, {val('gpt2_kv_seq_gb')} GB per sequence", ha="center", va="top",
            color=C["data"], fontsize=11, fontweight="bold", alpha=1.0 if k == 2 else 0.0)
    ax.set_xlim(0, 22)
    ax.set_ylim(-0.9, 1.7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Bytes read per decode step (GB; GPT-2 XL, {val('gpt2_ctx')}-token context, modeled)")
    sp.save(fig, out / f"decode_bytes_step{k}.svg")

# ------------------------------------------------- batch curve and TPOT cap
# Slot: 14.2 x 4.9 cm. Model and values from calcs.py (GPT-2 XL, MS Sec. 13.9.2).


def step_ms(b):
    """Step time in ms at batch b (a number or an array), from the scenario."""
    return np.array([K.m(K.CHAT.step_time(int(x)), "ms") for x in np.atleast_1d(b)]).reshape(np.shape(b))


for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=cm(14.2, 4.9))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.96, bottom=0.30)
    top = {1: 32, 2: K.B_TPOT, 3: K.B_MEM}[k]
    bs = np.arange(1, top + 1)
    ax.plot(bs / step_ms(bs) * 1000, step_ms(bs), color=C["memory"], lw=2.5)
    pts = [(K.B1, f"batch {K.B1}"), (K.B8, f"{K.B8}"), (K.B32, f"{K.B32}"), (K.B64, f"{K.B64}"),
           (K.B128, f"{K.B128}"), (K.B_TPOT, f"{K.B_TPOT}"), (K.B_MEM, "")]
    for b, lab in pts:
        if b > top:
            continue
        x, y = b / step_ms(b) * 1000, step_ms(b)
        derived = b > K.B32
        ax.plot([x], [y], "o", ms=6, color=C["neutral"] if derived else C["memory"],
                mec="white", zorder=5)
        ax.annotate(lab, (x, y), xytext=(-7, 3), textcoords="offset points",
                    ha="right", fontsize=9, fontweight="bold",
                    color=C["neutral"] if derived else C["memory"])
    if k >= 2:
        ax.axhline(K.TPOT_TARGET_MS, color=C["danger"], lw=2, ls="--")
        ax.text(200, 21, f"TPOT target {val('cs_tpot_target')} ms: at most batch {K.B_TPOT}", fontsize=9,
                color=C["danger"], fontweight="bold")
        ax.axvline(K.CEILING, color=C["neutral"], lw=1.5, ls=":")
        ax.text(K.CEILING + 150, 8, f"ceiling:\n{val('ceiling')} tokens/s",
                ha="left", va="bottom", fontsize=9, color=C["neutral"], fontweight="bold")
    if k >= 3:
        ax.text(K.CEILING + 150, 25.0, f"{K.B_MEM}: memory full", ha="left", fontsize=9,
                color=C["danger"], fontweight="bold")
    ax.set_xlim(0, 14200)
    ax.set_ylim(0, 30)
    ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000, 12000])
    ax.set_xticklabels(["0", "2K", "4K", "6K", "8K", "10K", "12K"])
    ax.tick_params(labelsize=8.5, pad=1)
    ax.set_xlabel("Aggregate tokens/s", fontsize=9, labelpad=1)
    ax.set_ylabel("TPOT (ms)", fontsize=9, labelpad=1)
    tag(ax, f"GPT-2 XL, one H100, full {val('gpt2_ctx')}-token contexts. Book: batch 1 and {K.B32}; "
        "gray: derived (bandwidth floor).", y=-0.30)
    for t in ax.texts[-1:]:
        t.set_fontsize(8)
    save_slot(fig, out / f"frontier_step{k}.svg")

# ------------------------------------------------ continuous batching grid
# Illustrative lengths (calcs.py: REQ1, REQ2). Slots: steps 1 to 3 at
# 14.2 x 6.0 cm, steps 4 to 7 at 14.2 x 4.9 cm.
REQ1 = [(n, K.LENGTHS[n]) for n in "ABCD"]
REQ2 = {n: K.LENGTHS[n] for n in "EFGH"}
LONGEST = max(L for _, L in REQ1)
END = K.SCHED.request_level_iterations
WAITING = "waiting:\n" + ", ".join(f"{n} {L}" for n, L in list(REQ2.items())[:2]) + ",\n" + \
          ", ".join(f"{n} {L}" for n, L in list(REQ2.items())[2:])


def grid(ax, title):
    for s in range(4):
        ax.axhspan(3 - s - 0.42, 3 - s + 0.42, xmin=0, xmax=1, color="#F4F4F4", lw=0, zorder=0)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(["slot 1", "slot 2", "slot 3", "slot 4"], fontsize=9)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, END)
    ax.set_ylim(-0.6, 3.6)
    ax.set_xticks(sorted({0, END, *(L for _, L in REQ1)}))
    ax.tick_params(axis="x", labelsize=8.5, pad=1)
    ax.set_xlabel("Decode step (lengths illustrative; one token per step, prefill omitted)",
                  fontsize=8.5, labelpad=1)
    ax.set_title(title, fontsize=9.5, loc="left", fontweight="bold", pad=3)


def bar(ax, slot, start, length, name, ck):
    ax.barh(3 - slot, length, left=start, height=0.7, color=C[ck], edgecolor="white", lw=1.5)
    ax.text(start + length / 2, 3 - slot, f"{name} {length}" if length >= 18 else name,
            ha="center", va="center", fontsize=9, color="white", fontweight="bold")


def idle(ax, slot, start, length):
    ax.barh(3 - slot, length, left=start, height=0.7, color=C["danger_l"],
            edgecolor=C["danger"], hatch="//", lw=0)


def summary(ax, text, color):
    ax.text(1.0, -0.30, text, ha="right", va="top", fontsize=9, fontweight="bold",
            color=color, transform=ax.transAxes)


used1 = K.SCHED.useful_iterations
used_all = K.SCHED.all_useful_iterations
for k in range(1, 8):
    h = 6.0 if k <= 3 else 4.9
    fig, ax = plt.subplots(figsize=cm(14.2, h))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.27 if k <= 3 else 0.31)
    if k <= 3:
        grid(ax, "Request-level: the batch is fixed until its longest member ends")
        for s, (n, L) in enumerate(REQ1):
            bar(ax, s, 0, L, n, "memory")
        if k == 1:
            idle(ax, 0, K.LENGTHS["A"], LONGEST - K.LENGTHS["A"])
        if k >= 2:
            for s, (n, L) in enumerate(REQ1):
                if L < LONGEST:
                    idle(ax, s, L, LONGEST - L)
            ax.text(55, 3, f"{used1} of {K.SCHED.slot_iterations} slot-steps used", ha="center", va="center",
                    fontsize=9, color=C["danger"], fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        if k == 3:
            for s, n in enumerate("EFGH"):
                bar(ax, s, K.REQ_SPAN[n][0], REQ2[n], n, "data")
            summary(ax, f"done at step {val('req_steps')}; {val('all_used')} of {val('req_total')} slot-steps used "
                    f"({val('req_util')}%)", C["danger"])
        else:
            ax.text(125, 1.5, WAITING, ha="center",
                    va="center", fontsize=9, color=C["data"], fontweight="bold")
    else:
        grid(ax, "Continuous: a freed slot takes the next request at the next step")
        for s, (n, L) in enumerate(REQ1):
            bar(ax, s, 0, L, n, "memory")
        if k == 4:
            ax.text(125, 1.5, WAITING, ha="center",
                    va="center", fontsize=9, color=C["data"], fontweight="bold")
        # Slots are figure layout; start and length come from the continuous schedule.
        cb = {n: (K.CB_SPAN[n][0], K.LENGTHS[n]) for n in "EFGH"}
        if k >= 5:
            bar(ax, 0, *cb["E"], "E", "data")
        if k >= 6:
            bar(ax, 2, *cb["F"], "F", "data")
            bar(ax, 0, *cb["G"], "G", "data")
            bar(ax, 0, *cb["H"], "H", "data")
            idle(ax, 2, K.CB_SPAN["F"][1], K.SCHED.continuous_iterations - K.CB_SPAN["F"][1])
            idle(ax, 3, K.CB_SPAN["D"][1], K.SCHED.continuous_iterations - K.CB_SPAN["D"][1])
        if k >= 7:
            summary(ax, f"done at step {val('cb_steps')}; {val('all_used')} of {val('cb_total')} slot-steps used "
                    f"({val('cb_util')}%)", C["data"])
    save_slot(fig, out / f"cb_step{k}.svg")

# ---------------------------------------------------------- stall timeline
# Slot: 9.9 x 4.4 cm. Derived in calcs.py (Llama 3 8B INT4, 32 decodes of 1,256 tokens).
DEC = K.DEC_MS
PRE = K.PRE_MS
for k in (1, 2, 3):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=cm(9.9, 4.4), sharey=True,
                                 gridspec_kw={"wspace": 0.10})
    fig.subplots_adjust(left=0.20, right=0.99, top=0.88, bottom=0.12)
    left = [DEC] * 4 + [DEC + PRE] + [DEC] * 4
    for i, v in enumerate(left):
        tall = i == 4
        a = 1.0 if (not tall or k >= 2) else 0.0
        ax.bar(i, v, color=C["compute"] if tall else C["memory"], width=0.75, alpha=a)
    ax.text(4, DEC + PRE + 1.5, f"{val('st_stall_ms')} ms", ha="center", fontsize=9,
            fontweight="bold", color=C["compute"], alpha=1.0 if k >= 2 else 0.0)
    ax.set_title("Whole prompt in one step", fontsize=8.5, loc="left",
                 fontweight="bold", pad=3)
    right = [DEC] * 2 + [DEC + PRE / K.CHUNKS] * K.CHUNKS + [DEC] * 3
    a3 = 1.0 if k >= 3 else 0.0
    for i, v in enumerate(right):
        chunk = 2 <= i <= 5
        bx.bar(i, v, color=C["compute"] if chunk else C["memory"], width=0.75, alpha=a3)
    bx.text(4.0, 27, f"{K.CHUNKS} chunks of {val('st_chunk_tok')}:\n{val('st_chunk_ms')} ms each", ha="center",
            fontsize=8.5, fontweight="bold", color=C["compute"], alpha=a3)
    bx.set_title("Four chunks", fontsize=8.5, loc="left", fontweight="bold",
                 alpha=a3, pad=3)
    for a_, show in ((ax, True), (bx, k >= 3)):
        a_.axhline(K.TPOT_TARGET_MS, color=C["danger"], lw=1.8, ls="--", alpha=1.0 if show else 0.0)
        a_.set_xticks([])
        a_.set_xlabel("decode steps", fontsize=8.5, alpha=1.0 if show else 0.0, labelpad=2)
        a_.set_ylim(0, 52)
        if not show:
            a_.spines["bottom"].set_alpha(0.0)
    ax.tick_params(axis="y", labelsize=8, pad=1)
    ax.set_yticks([0, K.TPOT_TARGET_MS, 2 * K.TPOT_TARGET_MS])
    ax.set_yticklabels(["0", f"TPOT {val('cs_tpot_target')}", f"{2 * K.TPOT_TARGET_MS:.0f}"])
    ax.get_yticklabels()[1].set_color(C["danger"])
    ax.get_yticklabels()[1].set_fontweight("bold")
    ax.text(0, DEC + 1.5, f"{val('st_dec_ms')} ms", ha="left", fontsize=8.5, color=C["memory"],
            fontweight="bold")
    ax.set_ylabel("Step time (ms)", fontsize=8.5, labelpad=1)
    save_slot(fig, out / f"stall_step{k}.svg")

# ------------------------------------------- KV budget, Llama 3 8B (derived)
# Slot: 8.7 x 4.6 cm. Values from calcs.py (MS Sec. 13.9.5).
ctx8 = np.linspace(0, 16384, 200)
fig, ax = plt.subplots(figsize=cm(8.7, 4.6))
fig.subplots_adjust(left=0.13, right=0.80, top=0.88, bottom=0.21)
FREE_GB = K.FREE_GB
ax.axhspan(FREE_GB, 125, color=C["danger_l"], alpha=0.6, lw=0)
ax.axhline(FREE_GB, color=C["danger"], lw=2, ls="--")
ax.text(16200, FREE_GB + 2, f"budget: {val('free_gb')} GB", ha="right", va="bottom",
        fontsize=9, color=C["danger"], fontweight="bold")
N8 = K.N8K
for b, ck in ((16, "data"), (32, "compute"), (N8, "network")):   # 16 and 32: illustrative batch sizes
    kv = b * ctx8 * K.L3_KV_TOK / 1e9
    ax.plot(ctx8, kv, color=C[ck], lw=2.5)
    if b < N8:
        ax.text(16600, b * 16384 * K.L3_KV_TOK / 1e9, f"{b} chats", ha="left",
                va="center", fontsize=9, color=C[ck], fontweight="bold")
ax.plot([K.CONV8.context_tokens], [N8 * K.PER_CONV_GB], "o", ms=7, color=C["network"], mec="white",
        mew=1.5, zorder=5)
ax.text(300, 97, f"{N8} chats:\nfull at {val('conv_tokens')} tokens", ha="left", va="top",
        fontsize=9, color=C["network"], fontweight="bold")
ax.set_xlim(0, 16384)
ax.set_ylim(0, 100)
ax.set_xticks([0, 4096, 8192, 12288, 16384])
ax.set_xticklabels(["0", "4K", "8K", "12K", "16K"])
ax.tick_params(labelsize=8, pad=1)
ax.set_xlabel("Context length (tokens)", fontsize=8.5, labelpad=1)
ax.set_ylabel("KV cache (GB, FP16)", fontsize=8.5, labelpad=1)
ax.set_title("Llama 3 8B, 16-bit weights, one H100", fontsize=9, fontweight="bold",
             loc="left", pad=3)
save_slot(fig, out / "kv_budget_8b.svg")

# ------------------------------------------- KV budget, 70B class (variant)
ctx = np.linspace(0, 32768, 200)
fig, ax = plt.subplots(figsize=cm(8.7, 4.6))
fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.21)
B70 = K.BUDGET70_GB
ax.axhspan(B70, 125, color=C["danger_l"], alpha=0.6, lw=0)
ax.axhline(B70, color=C["danger"], lw=2, ls="--")
ax.text(32000, B70 + 2.5, f"budget: {val('kv70_budget')} GB", ha="right", va="bottom",
        fontsize=8.5, color=C["danger"], fontweight="bold")
for b, ck in ((1, "memory"), (4, "data"), (16, "compute"), (32, "network")):
    kv = b * ctx * K.KV70_TOK / 1e9
    ax.plot(ctx, kv, color=C[ck], lw=2.5)
    if b > 4:
        xe = 102e9 / (b * K.KV70_TOK)
        lx, ly = xe * 0.96, 102
    else:
        lx = 24000 if b == 4 else 32000
        ly = b * lx * K.KV70_TOK / 1e9 + 3
    ax.text(lx, ly, f"batch {b}", ha="right", fontsize=8.5, color=C[ck], fontweight="bold")
x32 = K.X32
ax.plot([x32], [B70], "o", ms=7, color=C["network"], mec="white", mew=1.5, zorder=5)
ax.annotate(f"{val('kv70_batch')} chats full near\n{val('kv70_cross')} tokens", (x32, B70), xytext=(19500, 72),
            fontsize=8.5, color=C["network"], fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=C["network"]))
ax.set_xlim(0, 32768)
ax.set_ylim(0, 120)
ax.set_xticks([0, 8192, 16384, 24576, 32768])
ax.set_xticklabels(["0", "8K", "16K", "24K", "32K"])
ax.tick_params(labelsize=8, pad=1)
ax.set_xlabel("Context length (tokens)", fontsize=8.5, labelpad=1)
ax.set_ylabel("KV cache (GB, FP16)", fontsize=8.5, labelpad=1)
ax.set_title("Larger model: 70B class, 4-bit weights, one H100", fontsize=9,
             fontweight="bold", loc="left", pad=3)
save_slot(fig, out / "kv_budget.svg")

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
# Slot: 12.0 x 2.7 cm. Kwon et al., SOSP 2023, Fig. 2.
names = list(K.USED)
used = [K.USED[n] for n in names]
fig, ax = plt.subplots(figsize=cm(12.0, 2.7))
fig.subplots_adjust(left=0.17, right=0.97, top=0.98, bottom=0.30)
for i, n in enumerate(names):
    y = 3 - i
    ck = "data" if n == "vLLM" else "neutral"
    ax.barh(y, used[i], color=C[ck], height=0.66)
    ax.text(used[i] + 1.5, y, f"{used[i]:g}%", va="center", fontsize=8.5,
            fontweight="bold", color=C[ck])
    ax.text(-1.5, y, n, ha="right", va="center", fontsize=8.5, fontweight="bold")
ax.set_xlim(0, 110)
ax.set_yticks([])
ax.set_ylim(-0.5, 3.5)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="x", labelsize=8, pad=1)
ax.set_xlabel("Measured share of KV memory holding token states (%)", fontsize=8.5, labelpad=1)
save_slot(fig, out / "vllm_used.svg")

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
    ax.text(0.05, 0.5, f"Block size 4 for legibility (vLLM default: {val('block_default')})", fontsize=9,
            color=C["neutral"])
    ax.set_xlim(-0.2, 6.95)
    ax.set_ylim(0.3, 3.8)
    hide_axis(ax)
    sp.save(fig, out / f"paging_step{k}.svg")

# ------------------------------------------------------ prefix sharing
# Slot: 7.6 x 3.4 cm, drawn in centimeters. Sizes from calcs.py (2,000 tokens x 131,072 B).
COPIES = K.m(K.PREFIX.kv_unshared, "GB")
ONE = K.m(K.PREFIX.kv_shared, "GB")
for k in (1, 2):
    fig = plt.figure(figsize=cm(7.6, 3.4))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    for j in range(5):
        ax.add_patch(Rectangle((0.15 + j * 0.62, 2.15), 0.55, 0.55, facecolor=C["network_l"],
                               edgecolor=C["network"], lw=1.2))
    ax.text(0.15, 2.95, f"system prompt: {val('prefix_tokens')} tokens, {val('prefix_mb')} MB, stored once",
            ha="left", fontsize=8.5, color=C["network"], fontweight="bold")
    for i in range(3):
        x = 0.15 + i * 1.05
        ax.add_patch(Rectangle((x, 0.55), 0.9, 0.55, facecolor=C["memory_l"],
                               edgecolor=C["memory"], lw=1.2))
        ax.text(x + 0.45, 0.825, f"chat {i + 1}" if i < 2 else "chat N", ha="center",
                va="center", fontsize=8.5)
        ax.add_patch(FancyArrowPatch((x + 0.45, 1.12), (1.65 + (x - 1.2) * 0.4, 2.12),
                                     arrowstyle="-|>", mutation_scale=9,
                                     color=C["network"], lw=1.2))
    ax.text(0.15, 0.12, "each block table points at the shared blocks", ha="left",
            fontsize=8, color=C["neutral"])
    a = 1.0 if k >= 2 else 0.0
    bars = [(f"{val('prefix_users')} copies", COPIES, "danger", f"{val('prefix_copies_gb')} GB"),
            ("Shared", ONE, "data", f"{val('prefix_shared_gb')} GB")]
    for i, (lab, v, ck, txt) in enumerate(bars):
        y = 1.55 - i * 0.65
        ax.add_patch(Rectangle((4.75, y), v / COPIES * 1.4, 0.45, color=C[ck], alpha=a))
        ax.text(4.68, y + 0.22, lab, ha="right", va="center", fontsize=8.5,
                fontweight="bold", alpha=a)
        ax.text(4.75 + v / COPIES * 1.4 + 0.07, y + 0.22, txt, va="center", fontsize=8.5,
                fontweight="bold", color=C[ck], alpha=a)
    ax.text(7.45, 0.55, f"{val('prefix_users')} conversations\n(derived)", ha="right", fontsize=8,
            color=C["neutral"], alpha=a)
    ax.set_xlim(0, 7.6)
    ax.set_ylim(0, 3.4)
    hide_axis(ax)
    save_slot(fig, out / f"share_step{k}.svg")

# ------------------------------------------------------------ DistServe
# Slot: 9.3 x 3.4 cm. Zhong et al., OSDI 2024, Fig. 1 and Sec. 1 (measured).
fig, ax = plt.subplots(figsize=cm(9.3, 3.4))
fig.subplots_adjust(left=0.24, right=0.97, top=0.97, bottom=0.30)
DS = [("Both phases", K.COLO, "neutral"), ("Prefill only", K.PRE_ONLY, "compute"),
      ("Decode only", K.DEC_ONLY, "memory")]
for i, (lab, v, ck) in enumerate(DS):
    y = 2 - i
    ax.barh(y, v, color=C[ck], height=0.62)
    ax.text(v + 0.15, y, f"{v:g}", va="center", fontsize=9.5, fontweight="bold", color=C[ck])  # measured values as printed
    ax.text(-0.2, y, lab, ha="right", va="center", fontsize=9, fontweight="bold")
ax.set_xlim(0, 11.5)
ax.set_yticks([])
ax.set_ylim(-0.5, 2.5)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="x", labelsize=8, pad=1)
ax.set_xlabel(f"Requests/s per A100 meeting both deadlines\n({val('ds_params')}B model, measured)",
              fontsize=8.5, labelpad=1)
save_slot(fig, out / "distserve_bars.svg")

fig, ax = plt.subplots(figsize=(4.4, 2.3))
for i, (lab, ck) in enumerate((("prefill", "compute"), ("prefill", "compute"))):
    ax.add_patch(Rectangle((0.1, 1.3 - i * 1.0), 1.2, 0.7, facecolor=C[ck + "_l"],
                           edgecolor=C[ck], lw=1.8))
    ax.text(0.7, 1.65 - i * 1.0, f"GPU {i + 1}\n{lab}: {K.PRE_ONLY:g}/s", ha="center",
            va="center", fontsize=10.5, fontweight="bold", color=C[ck])
ax.add_patch(Rectangle((2.6, 0.8), 1.2, 0.7, facecolor=C["memory_l"],
                       edgecolor=C["memory"], lw=1.8))
ax.text(3.2, 1.15, f"GPU 3\ndecode: {K.DEC_ONLY:g}/s", ha="center", va="center", fontsize=10.5,
        fontweight="bold", color=C["memory"])
for yy in (1.65, 0.65):
    ax.add_patch(FancyArrowPatch((1.32, yy), (2.58, 1.15), arrowstyle="-|>",
                                 mutation_scale=12, color=C["network"], lw=1.6))
ax.text(1.95, 1.85, "KV cache", ha="center", fontsize=10, color=C["network"],
        fontweight="bold")
ax.text(2.0, -0.05, f"{val('ds_dec')} requests/s on {val('ds_gpus')} GPUs: {val('ds_per_gpu')} each, "
        f"{val('ds_gain')}× colocated", ha="center",
        fontsize=10.5, fontweight="bold", color=C["data"])
ax.set_xlim(0, 4.0)
ax.set_ylim(-0.3, 2.1)
hide_axis(ax)
sp.save(fig, out / "distserve_pool.svg")

sp.hbars(out / "split_link.svg", [
    (f"Cache traffic at {val('ds_kv_rate')} req/s", num("ds_traffic"), "network", f"{val('ds_traffic')} GB/s"),
    ("NVLink, A100", num("ds_link"), "memory", f"{val('ds_link')} GB/s"),
], "", log=False, size=(5.0, 1.5),
    xlabel=f"GB/s (OPT-66B, {val('ds_in')}-token prompts; DistServe)")

# ------------------------------------------- API prices (book figure 9 variant)
# Slide variant of MS figure 9, drawn from the same series the book uses:
# mlsysim/literature/data/series/llm_api_input_prices.csv (list price per
# million input tokens, with dated vendor sources). The case study's
# derived floor (calcs.py COST_M, MS Sec. 13.10.3) is drawn on it.
# Slot: 8.7 x 5.1 cm.
import csv  # noqa: E402
from datetime import date  # noqa: E402

PRICES = Path(__file__).resolve().parents[4] / "mlsysim" / "mlsysim" / "literature" / "data" / "series" / "llm_api_input_prices.csv"
rows = list(csv.DictReader(PRICES.open()))
OFFSET = {"Legacy GPT-3.5": (4, -11, "left"), "GPT-3.5 Turbo": (5, -10, "left"),
          "GPT-4 (Original)": (5, 2, "left"), "Claude 3 Opus": (5, 3, "left"),
          "Claude 3 Haiku": (-5, -2, "right"), "GPT-4o": (5, 2, "left"),
          "GPT-4o-mini": (5, -3, "left"), "Gemini 1.5 Flash": (-5, 3, "right"),
          "DeepSeek-V3": (-5, 5, "right")}
for k in (1, 2):
    fig, ax = plt.subplots(figsize=cm(8.7, 5.1))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.12)
    for r in rows:
        d = date.fromisoformat(r["date"])
        x = d.year + (d.timetuple().tm_yday - 1) / 365.0
        y = float(r["usd_per_mtok_input"])
        ax.plot([x], [y], "o", ms=5.5, color=C["memory"], mec="white", zorder=5)
        dx, dy, ha = OFFSET[r["model"]]
        ax.annotate(r["model"], (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=7.5, color=C["text"])
    if k >= 2:
        ax.axhline(K.COST_M, color=C["data"], lw=2.2)
        ax.text(2022.75, K.COST_M * 0.86, f"\\${val('cs_usd_m')}: Llama 3 8B\ncase-study bound (derived)",
                ha="left", va="top", fontsize=8, color=C["data"], fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.02, 80)
    ax.set_xlim(2022.7, 2025.4)
    ax.set_xticks([2023, 2024, 2025])
    ax.set_xticklabels(["2023", "2024", "2025"])
    ax.set_yticks([0.1, 1, 10])
    ax.set_yticklabels(["\\$0.10", "\\$1", "\\$10"])
    ax.tick_params(labelsize=8, pad=1)
    ax.set_ylabel("\\$ per M input tokens", fontsize=8.5, labelpad=1)
    save_slot(fig, out / f"api_price_step{k}.svg")

# ------------------------------------------------------ joules per token
# Values from calcs.py (1,000 W and 600 W, MS Napkin Math 13.5).
sp.hbars(out / "joules_token.svg", [
    (f"Batch 1, {val('p_node_kw')} kW", K.J1, "danger", f"{val('j1')} J per token"),
    (f"Batch 1, {val('p_low')} W", K.J1_LOW, "neutral", f"{val('j1_low')} J per token"),
    (f"Batch {val('batch_32')}, {val('p_node_kw')} kW", K.J32, "data", f"{val('j32')} J per token"),
], "", log=False, value_fmt="{:.2f}", size=sp.SHORT,
    xlabel="GPT-2 XL decode, joules per token (derived; gray: the chapter's low-use power)")

print("figures written")
