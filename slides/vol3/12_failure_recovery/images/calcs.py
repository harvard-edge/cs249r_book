#!/usr/bin/env python3
"""Every derived number in Lecture 12 (Failure Recovery), one block per slide.

Inputs are typed with their source. Sources:
  FR  = the rendered chapter, Failure Recovery (Volume III), dev text 2026-10-17.
  NM  = a Napkin Math notebook of that chapter (scenario values the chapter
        labels hypothetical; slides say "illustrative").
  Ex  = an end-of-chapter exercise of that chapter (hypothetical givens).
  SEC = U.S. SEC, Release No. 34-70694 (Knight Capital Americas LLC), 2013.
  Huang 2024 = Huang et al., "Large Language Models Cannot Self-Correct
        Reasoning Yet," ICLR 2024, Tables 2 and 3 (arXiv 2310.01798v2).
  Cemri 2025 = Cemri et al., "Why Do Multi-Agent LLM Systems Fail?" (MAST),
        arXiv 2503.13657v3, Figure 1.
  Huang OSDI 2022 = Huang et al., "Metastable Failures in the Wild,"
        OSDI 2022, Table 1.
  CrowdStrike PIR = CrowdStrike, Preliminary Post Incident Review, executive
        summary, July 24, 2024. Microsoft blog, July 20, 2024.

Run from this folder: python3 calcs.py   (prints "label: value")
make_figs.py imports these values so figures and slides agree.
"""


def show(label, value):
    print(f"  {label}: {value}")


# ------------------------------------------------------------- hook (SEC)
# SEC para. 15 to 17 and 27: 8 servers, one missed by the deployment; the
# 212 parent orders; 4 million executions, 397 million shares, about 45 min;
# $460 million realized loss.
KNIGHT_SERVERS = 8
KNIGHT_MISSED = 1
KNIGHT_MINUTES = 45
KNIGHT_LOSS_M = 460
KNIGHT_SHARES_M = 397


def knight():
    return dict(
        correct_servers=KNIGHT_SERVERS - KNIGHT_MISSED,          # 7 uninstalled
        loss_per_min_m=KNIGHT_LOSS_M / KNIGHT_MINUTES,            # $M per minute
        shares_per_min_m=KNIGHT_SHARES_M / KNIGHT_MINUTES,
    )


# --------------------------------------------- cleanup graph (Ex 1, FR 12.13)
# Durations once admitted (s): end test 30; preserve artifacts 20; release
# 40 (needs both); revoke credential 5 (needs release); scratch 10 (none).
CLEAN = dict(end_test=30, preserve=20, release=40, revoke=5, scratch=10)


def cleanup():
    c = CLEAN
    stack = sum(c.values())
    graph = max(max(c["end_test"], c["preserve"]) + c["release"] + c["revoke"],
                c["scratch"])
    return dict(stack_s=stack, graph_s=graph, saved_s=stack - graph)


# ---------------------------------------------- repair vs unwind (NM 12.1)
# FR 12.4.3: compensation 40.6 s; rebuild 46 s; rebuild tokens 38,500 in and
# 3,100 out at $3 and $15 per million; repair request 2,390 in, 180 out;
# 0.09 s pre-output, 50 tokens/s, 0.4 s validation; assumed p = 0.5.
P_IN, P_OUT = 3e-6, 15e-6


def repair():
    cf_t = 40.6 + 46.0
    cf_d = 38_500 * P_IN + 3_100 * P_OUT
    cr_t = 0.09 + 180 / 50 + 0.4
    cr_d = 2_390 * P_IN + 180 * P_OUT
    p = 0.5
    e_t = cr_t + (1 - p) * cf_t
    e_d = cr_d + (1 - p) * cf_d
    return dict(cf_t=cf_t, cf_d=cf_d, cr_t=cr_t, cr_d=cr_d,
                pstar_t=cr_t / cf_t, pstar_d=cr_d / cf_d,
                e_t=e_t, e_d=e_d, cut_t=1 - e_t / cf_t, cut_d=1 - e_d / cf_d,
                one_in=1 / 16)


# ------------------------------------------- self-correction (Huang 2024)
# GPT-3.5, Table 3 (intrinsic, round 2) and Table 2 (oracle labels).
HUANG = {
    "GSM8K": dict(standard=75.9, self_r2=74.7, oracle=84.3),
    "CommonSenseQA": dict(standard=75.8, self_r2=41.8, oracle=89.7),
    "HotpotQA": dict(standard=26.0, self_r2=25.0, oracle=29.0),
}


def huang():
    return {k: dict(drop=v["standard"] - v["self_r2"],
                    gain=v["oracle"] - v["standard"]) for k, v in HUANG.items()}


# ------------------------------------- CrowdStrike exposure (PIR, Microsoft)
# Published 04:09 UTC, reverted 05:27 UTC, 19 July 2024; 8.5 million devices.
def crowdstrike():
    return dict(window_min=(5 * 60 + 27) - (4 * 60 + 9))


# ---------------------------------- expected irreversible loss (NM 12.2)
# FR 12.5.2: p_bad 4 percent, consequence $12,000, catch 80 percent,
# revert fidelity 90 percent, attempt cost $20.
P_BAD, CONSEQ, CATCH, FIDELITY, ATTEMPT = 0.04, 12_000, 0.80, 0.90, 20


def pivot_loss(catch=CATCH):
    first = P_BAD * CONSEQ
    caught = P_BAD * catch * CONSEQ * (1 - FIDELITY)
    missed = P_BAD * (1 - catch) * CONSEQ
    total = caught + missed
    return dict(rollout_first=first, caught=caught, missed=missed, total=total,
                ratio=first / total, missed_share=missed / total,
                per_10pp=P_BAD * CONSEQ * 0.10 * FIDELITY,
                per_10pp_vs_attempt=P_BAD * CONSEQ * 0.10 * FIDELITY / ATTEMPT,
                total_vs_attempt=total / ATTEMPT)


# --------------------------------------------- loop cost (NM 12.3)
# FR 12.6.2: first input 1,000 tokens; each turn appends 1,100 (300 output +
# 800 observation); 300 output tokens per turn; 10 s per turn; $3 and $15 per
# million; ceiling 50 turns, stop after turn 4.
BASE, APPEND, OUT, TURN_S = 1_000, 1_100, 300, 10


def prompt_tokens(h):
    return h * BASE + APPEND * h * (h - 1) // 2


def loop_cost(h):
    return prompt_tokens(h) * P_IN + h * OUT * P_OUT


def loop():
    hi, lo = 50, 4
    return dict(ceiling_in=prompt_tokens(hi), stop_in=prompt_tokens(lo),
                ceiling_usd=loop_cost(hi), stop_usd=loop_cost(lo),
                ceiling_min=hi * TURN_S / 60, stop_s=lo * TURN_S,
                ratio=loop_cost(hi) / loop_cost(lo),
                saved_usd=loop_cost(hi) - loop_cost(lo),
                turn25_ratio=prompt_tokens(hi) / prompt_tokens(25),
                largest_call=BASE + APPEND * (hi - 1) + OUT)


# -------------------------------------------- loop data (Cemri 2025, MAST)
# Figure 1 prevalence, 1,642 traces, 7 frameworks: the five largest modes.
MAST = [("Step repetition", 15.7), ("Reasoning-action mismatch", 13.2),
        ("Unaware of termination", 12.4), ("Disobey task specification", 11.8),
        ("Incorrect verification", 9.1)]


# --------------------------------------------- retry layers (FR 12.7.2)
def retries(layers=3, attempts=3):
    return dict(layered=attempts ** layers, single=attempts)


# --------------------------------------------- capacity and jitter (NM 12.4)
# 256 blocking calls, 300 workers; cap 64 occupied + 32 queued; 128 retries
# in a 2 ms burst or spread over 2 s.
def jitter():
    burst = 128 / 0.002
    spread = 128 / 2.0
    return dict(free_uncapped=300 - 256, rejected=256 - (64 + 32),
                free_capped=300 - 64, burst_rate=burst, mean_rate=spread,
                ratio=burst / spread, expected_in_2ms=128 * 0.002 / 2.0)


# --------------------------------------- metastable incidents (OSDI 2022)
# Table 1: 21 incidents; sustaining effect "retry" listed for 12 of them;
# duration in hours as printed (SPF1 has no duration).
RETRY_SUSTAINED = [("IBM1", 73.53), ("AWS1", 66.7), ("GGL2", 21.5),
                   ("AWS4", 9.3), ("SPF2", 8.33), ("AZR4", 7.92),
                   ("AWS3", 4.55), ("AWS2", 4.23), ("GGL3", 3.2),
                   ("AZR2", 1.15), ("CAS2", 0.16)]
N_INCIDENTS, N_RETRY = 21, 12


def metastable():
    hrs = sorted(h for _, h in RETRY_SUSTAINED)
    return dict(share=N_RETRY / N_INCIDENTS, median_h=hrs[len(hrs) // 2],
                over_4h=sum(h > 4 for h in hrs))


# ------------------------------------------ breaker window (Ex 4, FR 12.13)
# 50 outcomes: 14 timeouts, 6 server errors, 12 endpoint validation
# rejections, 3 authorization denials, 15 successes; n_min 20, r_trip 0.5.
def breaker():
    n = 14 + 6 + 12 + 3 + 15
    classified = (14 + 6) / n
    naive = (n - 15) / n
    return dict(n=n, classified=classified, naive=naive)


if __name__ == "__main__":
    print("hook (SEC):")
    for k, v in knight().items():
        show(k, round(v, 2))
    print("cleanup graph (Ex 1):")
    for k, v in cleanup().items():
        show(k, v)
    print("repair vs unwind (NM 12.1):")
    r = repair()
    show("C_fallback time s", round(r["cf_t"], 2))
    show("C_fallback $", round(r["cf_d"], 4))
    show("C_repair time s", round(r["cr_t"], 2))
    show("C_repair $", round(r["cr_d"], 5))
    show("p* time %", round(100 * r["pstar_t"], 1))
    show("p* dollars %", round(100 * r["pstar_d"], 1))
    show("E at p=0.5 s", round(r["e_t"], 2))
    show("E at p=0.5 $", round(r["e_d"], 5))
    show("time cut %", round(100 * r["cut_t"], 1))
    show("dollar cut %", round(100 * r["cut_d"], 1))
    show("one in 16 %", round(100 * r["one_in"], 2))
    print("self-correction (Huang 2024, GPT-3.5):")
    for k, v in huang().items():
        show(k, {a: round(b, 1) for a, b in v.items()})
    print("CrowdStrike:")
    show("window min", crowdstrike()["window_min"])
    print("pivot loss (NM 12.2):")
    for k, v in pivot_loss().items():
        show(k, round(v, 3))
    print("loop (NM 12.3):")
    for k, v in loop().items():
        show(k, round(v, 3))
    print("retries:")
    show("layers", retries())
    print("jitter (NM 12.4):")
    for k, v in jitter().items():
        show(k, round(v, 3))
    print("metastable (OSDI 2022):")
    for k, v in metastable().items():
        show(k, round(v, 3))
    print("breaker (Ex 4):")
    for k, v in breaker().items():
        show(k, round(v, 3))
