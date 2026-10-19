"""Lecture 17 (Reinforcement Learning from Verifiable Rewards): every derived number.

Run: python3 calcs.py
Each block names the slide that shows its values, the inputs with their
sources, and prints the values with the labels used on the slide. Inputs typed
here come straight from a source (the chapter, a paper's table, our own runs in
data/); everything printed is derived. make_figs.py imports these functions.
"""

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def show(label, value, fmt="{:.4g}"):
    print(f"  {label}: {fmt.format(value)}")


# ---------------------------------------------------------------- hook (METR)
# Von Arx, Chan, Barnes, "Recent Frontier Models Are Reward Hacking", METR blog,
# 2025-06-05, task table (o3) and prompt table (Optimize LLM Foundry, 20 runs).
METR_TASKS = [  # (task, hacks, runs)
    ("Optimize LLM Foundry", 21, 21),
    ("Rust Codecontest scaffold", 12, 28),
    ("Optimize a Kernel", 6, 24),
    ("RE-Bench, all tasks", 39, 128),
    ("HCAST, all tasks", 8, 1087),
]
METR_PROMPTS = [  # (appended instruction, percent hacking as published)
    ("Original prompt", 80),
    ("“Only methods the designer intended”", 95),
    ("“Please do not cheat”", 80),
    ("“Please do not reward hack”", 70),
    ("“Researchers on Alzheimer's rely on this”", 70),
]


def metr():
    return [(t, 100 * h / n) for t, h, n in METR_TASKS]


# ---------------------------------------------------------------- baselines
# Chapter Sec. 17.2.2: fixed baseline 0.25 on a binary reward.
def baseline_scores(b=0.25):
    return 1 - b, 0 - b


# ---------------------------------------------------------------- NM 17.1
# Chapter NM 17.1: 8 rollouts x 1,024 tasks, pass rate 10%, verifier accepts every
# success and 2.5% of failures. Checkpoint 17.1 asks the share at 5%.
def false_share(p, fa=0.025, ta=1.0):
    return (1 - p) * fa / (p * ta + (1 - p) * fa)


def nm171():
    n = 8 * 1024
    true_acc = n * 0.10 * 1.0
    false_acc = n * 0.90 * 0.025
    return n, true_acc, false_acc, false_acc / (true_acc + false_acc)


# ---------------------------------------------------------------- shaping
# Chapter Sec. 17.3.3: R = c + alpha*f - lambda*len; order holds if alpha + lambda < 1.
# Exercise 2 inputs: alpha 0.4, lambda 0.7, cap 8,000 tokens.
def shaped(c, f, tokens, alpha, lam, cap=8000):
    return c + alpha * f - lam * tokens / cap


# ---------------------------------------------------------------- credit
# Chapter Sec. 17.4.1: baseline 0.4 on a failed rollout. Sec. 17.4.3: forks of 20,
# 1 pass then 12 passes. NM 17.2: 30 turns x 512 tokens, 8 continuations.
def forks(m=20, before=1, after=12):
    v0, v1 = before / m, after / m
    se = 1 / (2 * math.sqrt(m))
    se_diff = math.sqrt(2) / (2 * math.sqrt(m))
    return v0, v1, v1 - v0, se, se_diff, 2 * se_diff


def nm172(H=30, toks=512, m=8, boundaries=None):
    base = H * toks
    ks = range(1, H) if boundaries is None else boundaries
    branches = m * len(list(ks))
    total = base + m * toks * sum(H - k for k in ks)
    return base, branches, total, total / base


# ---------------------------------------------------------------- GRPO
# Chapter equation 2 (population std, divisor G) and NM 17.3; TRL GRPOTrainer
# (commit 88365e5) uses nanstd with Bessel's correction and a 1e-4 stabilizer.
def grpo(rewards, eps=1e-8, bessel=False):
    g = len(rewards)
    mu = sum(rewards) / g
    var = sum((r - mu) ** 2 for r in rewards) / (g - 1 if bessel else g)
    sd = math.sqrt(var)
    return mu, sd, [(r - mu) / (sd + eps) for r in rewards]


def lone_pass(G, eps=0.0):
    return (G - 1) / (math.sqrt(G - 1) + G * eps)


# ---------------------------------------------------------------- contrast
# Chapter Sec. 17.5.4 and NM 17.4: P(contrast) = 1 - p^G - (1-p)^G.
def contrast(p, G):
    return 1 - p ** G - (1 - p) ** G


def nm174():
    tokens_step = 512 * 8 * 30 * 512
    rows = [("Current task mix", 8, 0.05), ("Double the group", 16, 0.05), ("Target difficulty", 8, 0.50)]
    out = [(name, G, p, contrast(p, G), G / contrast(p, G)) for name, G, p in rows]
    return tokens_step, out


# ---------------------------------------------------------------- all-pass rule
# Chapter NM 17.5: incorrect patch 0.20 per trial, correct 0.98, 3 trials, group of 16.
def allpass(q_bad=0.20, q_good=0.98, n=3, G=16):
    return (1 - (1 - q_bad) ** G, q_bad ** n, 1 - q_good ** n,
            sum(q_bad ** j for j in range(n)), sum(q_good ** j for j in range(n)))


# ---------------------------------------------------------------- clipping
# Chapter Sec. 17.8.3 and DAPO Sec. 3.1: epsilon 0.2, tokens at 0.9 and 0.01.
def clip_room(p, eps=0.2):
    return p * (1 + eps), p * eps


# ---------------------------------------------------------------- length bias
# Chapter Sec. 17.5.3: failures of 128 and 4,096 tokens with the same coefficient.
def length_ratio(short=128, long=4096):
    return long / short


# ---------------------------------------------------------------- slot cycle
# Chapter NM 17.6: 256 slots, 20 decisions x (120 ms model + 80 ms tool), 0.5 s
# assessment, cold reset 3.5 s, snapshot 15 ms.
def nm176(slots=256, H=20, model=0.120, tool=0.080, assess=0.5, cold=3.5, snap=0.015):
    work = H * (model + tool) + assess
    c_cold, c_snap = work + cold, work + snap
    cap_cold, cap_snap = slots * 3600 / c_cold, slots * 3600 / c_snap
    return work, c_cold, c_snap, cap_cold, cap_snap, cap_snap / cap_cold, cold / snap, work / c_cold


# ---------------------------------------------------------------- verifier pool
# Chapter Sec. 17.9.5: 512 trajectories every 16 s, 3 trials of 8 s, 80% target.
def pool(arrivals=512, every=16, trials=3, t=8, target=0.8):
    lam = arrivals / every
    one = lam * t
    busy = lam * trials * t
    return lam, one, busy, math.ceil(busy / target)


# ---------------------------------------------------------------- logprobs (our run)
# data/logprob_run.json: Ollama 0.34.4, gemma3:1b, top-3 log probabilities as reported.
def logprob_run():
    d = json.loads((HERE / "data" / "logprob_run.json").read_text())
    runs = {r["temperature"]: r for r in d["runs"]}
    tok = runs[0.2]["tokens"][2]       # third sampled token at T = 0.2
    top = tok["top"]
    p_raw = math.exp(tok["logprob"])
    # probability of the same token after temperature 0.2, renormalized over the
    # reported top 3 (the omitted tail is negligible after raising to the 5th power)
    w = [math.exp(lp / 0.2) for _, lp in top]
    p_t = math.exp(tok["logprob"] / 0.2) / sum(w)
    return d, tok, p_raw, p_t


# ---------------------------------------------------------------- Yue et al. 2025
# Yue et al., arXiv 2504.13837, Table 3, Omni-MATH-Train, Qwen2.5-7B and GRPO steps.
YUE = [(0, 9.9, 67.2), (150, 26.1, 66.3), (300, 33.6, 65.3), (450, 42.5, 64.3)]


if __name__ == "__main__":
    print("Hook (METR task table): percent of runs that reward hacked")
    for t, pct in metr():
        show(t, pct, "{:.1f}%")

    print("Fixed baseline 0.25 (Sec. 17.2.2)")
    up, down = baseline_scores()
    show("pass coefficient", up, "{:+.2f}")
    show("fail coefficient", down, "{:+.2f}")

    print("False accepts in the reward (NM 17.1 and the estimate frame)")
    n, ta, fa, share = nm171()
    show("rollouts", n, "{:,}")
    show("true accepts", ta, "{:.2f}")
    show("false accepts", fa, "{:.2f}")
    show("false share at 10%", 100 * share, "{:.1f}%")
    show("false share at 5% (Checkpoint 17.1)", 100 * false_share(0.05), "{:.1f}%")
    show("false share at 40%", 100 * false_share(0.40), "{:.1f}%")

    print("Shaping order (Exercise 2: alpha 0.4, lambda 0.7)")
    show("pass, 7,200 tokens, f 0.5", shaped(1, 0.5, 7200, 0.4, 0.7), "{:.3f}")
    show("fail, 400 tokens, f 1", shaped(0, 1.0, 400, 0.4, 0.7), "{:.3f}")
    show("lowest pass 1 - lambda", 1 - 0.7, "{:.2f}")
    show("highest fail alpha", 0.4, "{:.2f}")
    show("alpha + lambda", 0.4 + 0.7, "{:.2f}")

    print("Credit (Sec. 17.4.1 baseline 0.4; Sec. 17.4.3 forks of 20)")
    show("failed rollout coefficient, baseline 0.4", 0 - 0.4, "{:+.1f}")
    v0, v1, d, se, sed, thr = forks()
    show("value before debugger", v0, "{:.2f}")
    show("value after debugger", v1, "{:.2f}")
    show("difference", d, "{:.2f}")
    show("SE bound per fork", se, "{:.2f}")
    show("SE bound of difference", sed, "{:.2f}")
    show("resolution threshold (2 x SE)", thr, "{:.2f}")
    b, br, tot, mult = nm172()
    show("base tokens", b, "{:,}")
    show("exhaustive branches", br, "{:,}")
    show("exhaustive tokens", tot, "{:,}")
    show("exhaustive multiple", mult, "{:.0f}x")
    b, br, tot, mult = nm172(boundaries=[10, 20])
    show("two-fork branches", br, "{:,}")
    show("two-fork tokens", tot, "{:,}")
    show("two-fork multiple", mult, "{:.0f}x")

    print("GRPO coefficients (NM 17.3; TRL convention)")
    mu, sd, a = grpo([0, 1, 0])
    show("mean", mu, "{:.6f}")
    show("std (divisor G)", sd, "{:.6f}")
    show("failure coefficient", a[0], "{:.6f}")
    show("pass coefficient", a[1], "{:.6f}")
    mu, sd, a = grpo([0, 1, 0], eps=1e-4, bessel=True)
    show("TRL std (divisor G-1)", sd, "{:.4f}")
    show("TRL failure coefficient", a[0], "{:.3f}")
    show("TRL pass coefficient", a[1], "{:.3f}")
    show("lone pass, G=3", lone_pass(3), "{:.2f}")
    show("lone pass, G=16", lone_pass(16), "{:.2f}")
    _, _, a16 = grpo([1] + [0] * 15, eps=0)
    show("each failure, G=16", a16[1], "{:.2f}")
    show("per-token penalty ratio 128 vs 4,096", length_ratio(), "{:.0f}x")

    print("Zero-contrast groups (Sec. 17.5.4, NM 17.4, Checkpoint 17.2)")
    show("G=2, p=0.10 contrast", contrast(0.10, 2), "{:.2f}")
    show("G=8, p=0.05 zero-contrast share", 100 * (1 - contrast(0.05, 8)), "{:.0f}%")
    tokens_step, rows = nm174()
    show("tokens per step", tokens_step / 1e6, "{:.1f} M")
    for name, G, p, c, r in rows:
        show(f"{name} (G={G}, p={p}) contrast", c, "{:.2f}")
        show(f"{name} rollouts per informative group", r, "{:.1f}")
    show("targeting saving", rows[0][4] / rows[2][4], "{:.1f}x")
    show("Checkpoint 17.2: G=8, p=0.20 contrast", contrast(0.20, 8), "{:.2f}")
    show("Checkpoint 17.2: rollouts per informative group", 8 / contrast(0.20, 8), "{:.1f}")

    print("All-pass rule (NM 17.5)")
    g, acc_bad, rej_good, e_bad, e_good = allpass()
    show("group of 16 incorrect, P(at least one false pass)", 100 * g, "{:.1f}%")
    show("incorrect accepted after 3 trials", 100 * acc_bad, "{:.1f}%")
    show("correct rejected after 3 trials", 100 * rej_good, "{:.2f}%")
    show("expected trials, incorrect", e_bad, "{:.2f}")
    show("expected trials, correct", e_good, "{:.4f}")

    print("Clipping room (Sec. 17.8.3, epsilon 0.2)")
    for p in (0.9, 0.01):
        hi, room = clip_room(p)
        show(f"token at {p}: ceiling", hi, "{:.3f}")
        show(f"token at {p}: room", room, "{:.3f}")

    print("Slot cycle (NM 17.6)")
    w, cc, cs, kc, ks, ratio, rs, frac = nm176()
    show("non-reset work (s)", w, "{:.2f}")
    show("cold cycle (s)", cc, "{:.3f}")
    show("snapshot cycle (s)", cs, "{:.3f}")
    show("cold capacity (episodes/h)", kc, "{:,.0f}")
    show("snapshot capacity (episodes/h)", ks, "{:,.0f}")
    show("capacity ratio", ratio, "{:.2f}x")
    show("reset speedup", rs, "{:.0f}x")
    show("non-reset share, cold", 100 * frac, "{:.2f}%")
    show("reset share, cold", 100 * (1 - frac), "{:.0f}%")

    print("Verifier pool (Sec. 17.9.5)")
    lam, one, busy, slots = pool()
    show("arrival rate (trajectories/s)", lam, "{:.0f}")
    show("busy workers, one trial", one, "{:.0f}")
    show("busy workers, three trials", busy, "{:.0f}")
    show("slots at 80% target", slots, "{:.0f}")

    print("Logprob run (data/logprob_run.json)")
    d, tok, p_raw, p_t = logprob_run()
    print("  server:", d["server"])
    print(f"  token at T=0.2: {tok['token']!r}, reported logprob {tok['logprob']:.3f}")
    show("reported probability", p_raw, "{:.3f}")
    show("probability under T=0.2 sampling (top-3 renormalized)", p_t, "{:.3f}")
    show("ratio reported / sampled", p_raw / p_t, "{:.1f}x")
    alt = [t for t in d["runs"] if t["temperature"] == 1.5][0]["tokens"][2]
    print(f"  token at T=1.5: {alt['token']!r}, probability {100 * math.exp(alt['logprob']):.1f}%")

    print("Yue et al. 2025, Table 3 (Omni-MATH-Train)")
    for step, p1, p256 in YUE:
        print(f"  step {step}: pass@1 {p1}%, pass@256 {p256}%")
    show("pass@1 change", YUE[-1][1] - YUE[0][1], "{:+.1f} pts")
    show("pass@256 change", YUE[-1][2] - YUE[0][2], "{:+.1f} pts")
