#!/usr/bin/env python3
"""Every derived number in Lecture 4 (Context Engineering), one block per slide.

"CE" is the rendered chapter, Context Engineering (Volume III, Chapter 4).
Inputs carry their source as a comment; everything printed is derived here
and copied onto the slide and into its note. Scenario inputs from the
chapter's Napkin Math (NM) notebooks are hypothetical, so slides that show
them say "illustrative".

Run from anywhere:  python3 calcs.py
make_figs.py imports the functions below, so figures and slides agree.
"""

import math
import sys
from pathlib import Path

MLSYSIM = Path(__file__).resolve().parents[4] / "mlsysim"


# ------------------------------------------------------------------ inputs
# NM 4.1 (CE Sec. 4.2): the repair harness, scenario values.
WINDOW = 128_000        # S_max, context window
RESERVE = 8_000         # K_max, output reserve
SYSTEM = 3_000          # system instructions
TOOLS = 12_000          # tool declarations
STATE = 24_000          # task state
OBS_TURN = 4_000        # appended tool result per turn
TURNS = 30              # step budget
DIGEST = 2_000          # compacted digest per turn (NM 4.1 insight)
LEAF = 8_000            # selected observations per call (NM 4.2, Sec. 4.4.1)

# NM 4.2 (CE Sec. 4.4.3): prices are hypothetical.
PRICE = 3.0 / 1e6       # $ per uncached input token
CACHED = 0.10           # cached reads billed at 10 percent
TRUNK_REWRITES = 2      # compaction passes in the step budget (NM 4.3)

# NM 4.3 (CE Sec. 4.5.3)
TARGET = 47_000         # post-compaction target
TRIGGER = 92_000        # trigger watermark
TARGET_NARROW = 81_000  # the narrower variant

# NM 4.4 (CE Sec. 4.7.3)
STALE = 850             # outdated r17 excerpt at the start of the trunk
TOMBSTONE = 24          # in-place marker
NOTICE = 128            # appended notice with framing

# Equation 2 toy head (CE Sec. 4.3.1)
TARGET_LOGIT = 6.0
BACKGROUND_LOGIT = -2.0
LOOKALIKE_LOGIT = 2.0   # Exercise 3: superseded excerpt keys
EX3_UNRELATED = 40_000  # Exercise 3

# Equation 6 (CE Sec. 4.8.2): hypothetical per-round loss
LOSS = 0.05

# Exercise 6 (CE Sec. 4.12), hypothetical
EX6_CONSTRAINTS = 12
EX6_LOSS = 0.03
EX6_ROUNDS = 3
EX6_PACC = 0.6
EX6_ATTEMPT = 2.00
EX6_DETECT = 0.90
EX6_CHECK = 0.05        # $ per round

# Lecture 2 recall (Foundation Model NM 2.2) and Lecture 1 (NM 1.2)
L2_FIRST, L2_APPEND, L2_TURNS = 8_000, 1_500, 30
L1_ATTEMPT, L1_PACC = 2.10, 0.40


def p(label, value):
    print(f"{label:<58} {value}")


# ------------------------------------------------------------------ hook / NM 4.1
def budget():
    allowance = WINDOW - RESERVE
    after_fixed = allowance - SYSTEM - TOOLS
    obs_budget = after_fixed - STATE
    turns_fit = obs_budget // OBS_TURN
    return dict(
        allowance=allowance,
        after_fixed=after_fixed,
        after_fixed_share=after_fixed / WINDOW,
        obs_budget=obs_budget,
        turns_fit=turns_fit,
        overflow_turn=turns_fit + 1,
        max_per_turn=obs_budget / TURNS,
        digest_turns=obs_budget // DIGEST,
    )


def append_vs_select():
    fixed = SYSTEM + TOOLS + STATE
    append = [fixed + OBS_TURN * t for t in range(1, TURNS + 1)]
    select = [fixed + LEAF for _ in range(1, TURNS + 1)]
    cum_a, cum_s, a, s = [], [], 0, 0
    for x, y in zip(append, select):
        a += x
        s += y
        cum_a.append(a)
        cum_s.append(s)
    return dict(per_call_append=append, per_call_select=select,
                cum_append=cum_a, cum_select=cum_s,
                ratio=cum_a[-1] / cum_s[-1],
                final_append_request=append[-1],
                final_with_reserve=append[-1] + RESERVE)


def recall_l2():
    resend = L2_TURNS * L2_FIRST + L2_APPEND * L2_TURNS * (L2_TURNS - 1) // 2
    fresh = L2_FIRST + (L2_TURNS - 1) * L2_APPEND
    return dict(resend=resend, fresh=fresh, ratio=resend / fresh,
                ceff=L1_ATTEMPT / L1_PACC)


# ------------------------------------------------------------------ usable context
def dilution(n_keys):
    """Target weight among n_keys total keys (target plus n_keys - 1 others)."""
    t = math.exp(TARGET_LOGIT)
    return t / (t + (n_keys - 1) * math.exp(BACKGROUND_LOGIT))


def lookalike():
    t = math.exp(TARGET_LOGIT)
    bg = EX3_UNRELATED * math.exp(BACKGROUND_LOGIT)
    look = STALE * math.exp(LOOKALIKE_LOGIT)
    return dict(
        without=t / (t + bg),
        with_excerpt=t / (t + bg + look),
        equivalent_unrelated=STALE * math.exp(LOOKALIKE_LOGIT - BACKGROUND_LOGIT),
        drop_half_unrelated=t / (t + bg / 2 + look),
        drop_excerpt=t / (t + bg),
    )


def fiction_live():
    sys.path.insert(0, str(MLSYSIM))
    import numpy as np
    from mlsysim import Literature
    lengths = ("0", "400", "1k", "2k", "4k", "8k", "16k", "32k", "60k")
    cols = tuple("score_" + n for n in lengths)
    entries = Literature.LongContextComprehension.entries
    scored = [e for e in entries if all(e[c] is not None for c in cols)]
    arr = np.array([[e[c] for c in cols] for e in scored]) * 100
    return dict(lengths=lengths, n=len(scored),
                median=np.median(arr, axis=0), q25=np.percentile(arr, 25, axis=0),
                q75=np.percentile(arr, 75, axis=0),
                hold90=int((arr[:, -1] >= 90).sum()),
                lo60=arr[:, -1].min(), hi60=arr[:, -1].max())


# Liu et al. 2024 (TACL), Table 6: 20 documents (about 4K tokens), GPT-3.5-Turbo.
LIU_POS = [1, 5, 10, 15, 20]
LIU_ACC = [75.8, 57.2, 53.8, 55.4, 63.2]
LIU_CLOSED = 56.1   # Table 1, closed-book

# Hsieh et al. 2024 (COLM), RULER Table 3: claimed and effective length (K tokens).
# Effective "<4K" is drawn at 2K and labeled "<4K".
RULER = [("GPT-4", 128, 64), ("Llama3.1 (70B)", 128, 64), ("Qwen2 (72B)", 128, 32),
         ("Llama3.1 (8B)", 128, 32), ("Mixtral-8x22B", 64, 32),
         ("Mistral-v0.2 (7B)", 32, 16), ("DBRX", 32, 8), ("Together (7B)", 32, 4),
         ("LongChat (7B)", 32, None)]


def liu():
    return dict(drop_first_to_min=LIU_ACC[0] - min(LIU_ACC),
                below_closed=min(LIU_ACC) < LIU_CLOSED)


def ruler():
    tested = len(RULER)
    reached = sum(1 for _, c, e in RULER if e is not None and e >= c)
    return dict(tested=tested, reached=reached)


# ------------------------------------------------------------------ layout / NM 4.2
def layout_bill(rewrites=TRUNK_REWRITES):
    root = SYSTEM + TOOLS
    request = root + STATE + LEAF
    total = request * TURNS
    warm = TURNS - 1 - rewrites
    uncached = request + rewrites * (STATE + LEAF) + warm * LEAF
    cached = total - uncached
    unstable = total * PRICE
    stable = uncached * PRICE + cached * PRICE * CACHED
    return dict(root=root, request=request, total=total, uncached=uncached,
                cached=cached, hit=cached / total, unstable=unstable,
                stable=stable, ratio=unstable / stable,
                prefill_ratio=total / uncached,
                per_call_unstable=unstable / TURNS, per_call_stable=stable / TURNS,
                rewrite_tokens=STATE + LEAF,
                rewrite_in_leaf_calls=(STATE + LEAF) / LEAF)


def layout_every_fourth():
    """Checkpoint 4.1 variant: trunk rewritten on every fourth call after the first."""
    rewrites = (TURNS - 1) // 4
    return dict(rewrites=rewrites, **layout_bill(rewrites))


# SGLang (Zheng et al., NeurIPS 2024; arXiv 2312.07104v2, Sec. 6): one month in a
# public deployment, cache hit rate 52.4 percent (LLaVA-Next-34B) and 74.1
# percent (Vicuna-33B); first-token latency 1.7 times lower for Vicuna-33B.
SGLANG = [("LLaVA-Next-34B", 52.4), ("Vicuna-33B", 74.1)]


# ------------------------------------------------------------------ compaction / NM 4.3
def watermark(target=TARGET):
    cycle = math.ceil((TRIGGER - target) / OBS_TURN)
    trigger_len = target + cycle * OBS_TURN
    passes, length = 0, target
    trace = []
    for _ in range(1, TURNS + 1):
        length += OBS_TURN
        trace.append(length)
        if length >= TRIGGER:
            passes += 1
            length = target
    return dict(cycle=cycle, trigger_len=trigger_len,
                overshoot=trigger_len - TRIGGER,
                with_reserve=trigger_len + RESERVE, passes=passes, trace=trace)


# Maynez et al. 2020 (ACL), Table 2: percent of 500 XSum summaries per system
# with at least one hallucinated span (intrinsic or extrinsic, I u E).
MAYNEZ = [("PTGen", 75.3), ("TConvS2S", 78.5), ("TranS2S", 79.3),
          ("BertS2S", 73.1), ("Gold reference", 76.9)]


def survival():
    first_below_half = next(n for n in range(1, 200) if (1 - LOSS) ** n < 0.5)
    return dict(s10=(1 - LOSS) ** 10, s40=(1 - LOSS) ** 40,
                below_half=first_below_half)


def check_pays():
    n = EX6_CONSTRAINTS * EX6_ROUNDS
    s_no = (1 - EX6_LOSS) ** n
    p_no = EX6_PACC * s_no
    c_no = EX6_ATTEMPT / p_no
    undetected = EX6_LOSS * (1 - EX6_DETECT)
    s_chk = (1 - undetected) ** n
    p_chk = EX6_PACC * s_chk
    attempt_chk = EX6_ATTEMPT + EX6_ROUNDS * EX6_CHECK
    c_chk = attempt_chk / p_chk
    return dict(s_no=s_no, p_no=p_no, c_no=c_no, undetected=undetected,
                s_chk=s_chk, p_chk=p_chk, attempt_chk=attempt_chk, c_chk=c_chk)


# ------------------------------------------------------------------ invalidation / NM 4.4
def invalidation():
    request = SYSTEM + TOOLS + STATE + LEAF
    root = SYSTEM + TOOLS
    removed = request - STALE
    replaced = removed + TOMBSTONE
    appended = request + NOTICE
    return dict(removed=removed, removed_suffix=removed - root,
                replaced=replaced, replaced_suffix=replaced - root,
                appended=appended, appended_suffix=NOTICE,
                ratio=(replaced - root) / NOTICE)


# Shi et al. 2023 (ICML), code-davinci-002 with chain-of-thought prompting:
# base problems 95.0 (Table 2); GSM-IC-4K with one irrelevant sentence, micro
# accuracy 72.4, with an instruction to ignore irrelevant text 77.8 (Table 3).
SHI = [("Base problems", 95.0), ("One irrelevant sentence added", 72.4),
       ("Plus instruction to ignore it", 77.8)]


def shi():
    return dict(drop=SHI[0][1] - SHI[1][1], recovered=SHI[2][1] - SHI[1][1])


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    b = budget()
    print("== Hook and NM 4.1 (illustrative)")
    p("input allowance S_max - K_max (tokens)", f"{b['allowance']:,}")
    p("after instructions and tools (tokens)", f"{b['after_fixed']:,}")
    p("  share of window", f"{100 * b['after_fixed_share']:.1f}%")
    p("observation budget (tokens)", f"{b['obs_budget']:,}")
    p("turns that fit at 4,000 per turn", b["turns_fit"])
    p("overflow turn", b["overflow_turn"])
    p("max observation per turn for 30 turns", f"{b['max_per_turn']:,.0f}")
    p("turns with a 2,000-token digest", b["digest_turns"])
    a = append_vs_select()
    p("append cumulative input (tokens)", f"{a['cum_append'][-1]:,}")
    p("select cumulative input (tokens)", f"{a['cum_select'][-1]:,}")
    p("append / select", f"{a['ratio']:.1f}x")
    p("final append request (tokens)", f"{a['final_append_request']:,}")
    p("final append request plus reserve (tokens)", f"{a['final_with_reserve']:,}")

    r = recall_l2()
    print("== Recall (Lectures 1 and 2)")
    p("L2 resent input (tokens)", f"{r['resend']:,}")
    p("L2 fresh prefill with reuse (tokens)", f"{r['fresh']:,}")
    p("L2 ratio", f"{r['ratio']:.1f}x")
    p("L1 cost per accepted task", f"${r['ceff']:.2f}")

    print("== Attention dilution (toy head, eq. 2)")
    p("target weight among 100 keys", f"{100 * dilution(100):.1f}%")
    p("target weight among 100,000 keys", f"{100 * dilution(100_000):.1f}%")
    lk = lookalike()
    p("Ex 3: 40,000 unrelated keys, no excerpt", f"{100 * lk['without']:.1f}%")
    p("Ex 3: plus 850 look-alike keys", f"{100 * lk['with_excerpt']:.1f}%")
    p("Ex 3: unrelated keys equal to 850 look-alikes", f"{lk['equivalent_unrelated']:,.0f}")
    p("Ex 3: drop half the unrelated keys", f"{100 * lk['drop_half_unrelated']:.1f}%")
    p("Ex 3: drop the excerpt", f"{100 * lk['drop_excerpt']:.1f}%")

    f = fiction_live()
    print("== Fiction.LiveBench via Epoch AI (MLSysIM)")
    p("models scored through 60K", f["n"])
    for L, m in zip(f["lengths"], f["median"]):
        p(f"  median at {L}", f"{m:.0f}%")
    p("models at or above 90% at 60K", f["hold90"])
    p("range at 60K", f"{f['lo60']:.0f} to {f['hi60']:.0f}%")
    lu = liu()
    p("Liu: first minus worst position (points)", f"{lu['drop_first_to_min']:.1f}")
    ru = ruler()
    p("RULER rows plotted / reaching claimed length", f"{ru['tested']} / {ru['reached']}")

    lb = layout_bill()
    print("== NM 4.2 layout (illustrative)")
    p("request tokens", f"{lb['request']:,}")
    p("trajectory input tokens", f"{lb['total']:,}")
    p("unstable bill", f"${lb['unstable']:.2f}")
    p("stable uncached tokens", f"{lb['uncached']:,}")
    p("stable cached tokens", f"{lb['cached']:,}")
    p("cache hit share", f"{100 * lb['hit']:.1f}%")
    p("stable bill", f"${lb['stable']:.2f}")
    p("unstable / stable bill", f"{lb['ratio']:.1f}x")
    p("uncached prefill ratio", f"{lb['prefill_ratio']:.1f}x")
    p("per call unstable / stable", f"${lb['per_call_unstable']:.3f} / ${lb['per_call_stable']:.3f}")
    p("trunk rewrite tokens, in leaf-only calls", f"{lb['rewrite_tokens']:,} = {lb['rewrite_in_leaf_calls']:.0f}")
    e4 = layout_every_fourth()
    p("Checkpoint 4.1: rewrites on every fourth call", e4["rewrites"])
    p("  uncached tokens", f"{e4['uncached']:,}")
    p("  stable bill", f"${e4['stable']:.2f}")
    p("  unstable / this", f"{e4['ratio']:.1f}x")

    print("== NM 4.3 watermarks (illustrative)")
    w = watermark()
    p("additions per cycle", w["cycle"])
    p("trigger length / overshoot", f"{w['trigger_len']:,} / {w['overshoot']:,}")
    p("trigger length plus reserve", f"{w['with_reserve']:,}")
    p("passes in 30 turns", w["passes"])
    wn = watermark(TARGET_NARROW)
    p("narrow: additions per cycle", wn["cycle"])
    p("narrow: passes in 30 turns", wn["passes"])
    p("frequency ratio", f"{w['cycle'] / wn['cycle']:.0f}x")

    print("== Survival and Exercise 6 (illustrative)")
    s = survival()
    p("survive 10 rounds at 5% loss", f"{100 * s['s10']:.1f}%")
    p("survive 40 rounds", f"{100 * s['s40']:.1f}%")
    p("first round below 50%", s["below_half"])
    c = check_pays()
    p("no check: all 12 survive 3 rounds", f"{100 * c['s_no']:.1f}%")
    p("no check: p_acc", f"{c['p_no']:.3f}")
    p("no check: C_eff", f"${c['c_no']:.2f}")
    p("check: undetected loss per round", f"{c['undetected']:.3f}")
    p("check: all survive", f"{100 * c['s_chk']:.1f}%")
    p("check: p_acc", f"{c['p_chk']:.3f}")
    p("check: attempt cost", f"${c['attempt_chk']:.2f}")
    p("check: C_eff", f"${c['c_chk']:.2f}")

    print("== NM 4.4 invalidation (illustrative)")
    iv = invalidation()
    p("remove: resident / uncached suffix", f"{iv['removed']:,} / {iv['removed_suffix']:,}")
    p("replace: resident / uncached suffix", f"{iv['replaced']:,} / {iv['replaced_suffix']:,}")
    p("append: resident / uncached suffix", f"{iv['appended']:,} / {iv['appended_suffix']:,}")
    p("replace suffix / append suffix", f"{iv['ratio']:.1f}x")
    sh = shi()
    p("Shi: drop with one irrelevant sentence (points)", f"{sh['drop']:.1f}")
    p("Shi: recovered by instruction (points)", f"{sh['recovered']:.1f}")
