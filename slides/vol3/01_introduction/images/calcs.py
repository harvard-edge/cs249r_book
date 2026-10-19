#!/usr/bin/env python3
"""Every derived number in Lecture 1 (Introduction), computed from sourced inputs.

Run from this folder: python3 calcs.py
Each block prints the values with the label the slide or note uses. The deck's
make_figs.py imports this module, so a figure and its slide share one value.

Sources
  CH   the rendered chapter, Introduction (Volume III), dev text of 2026-10-17:
       Napkin Math 1.1 and 1.2, Lighthouse 1.1, Checkpoint 1.1, section 1.9.
  RUN  one public agent run on SWE-bench Verified: mini-swe-agent 1.0.0 with
       Claude 4 Sonnet (20250514), bash only, one attempt per task; reduced by
       extract_run.py to data/swebench_verified_mini_sonnet4.csv.
  METR Kwa et al. 2025, Time Horizon v1.1, through MLSysIM
       (Literature.AgentTimeHorizon), fitted as the chapter fits it.
  SWEB SWE-bench Verified record for astropy__astropy-14995 (Hugging Face
       dataset princeton-nlp/SWE-bench_Verified): one FAIL_TO_PASS test and
       179 PASS_TO_PASS tests.
"""

import csv
import datetime as dt
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MLSYSIM = Path(__file__).resolve().parents[4] / "mlsysim"


def show(label, value):
    print(f"  {label}: {value}")


# ----------------------------------------------------------------- NM 1.1
# CH Napkin Math 1.1: 99 percent per step, independent, unchecked.
STEP_OK = 0.99
H_SHORT, H_LONG = 10, 100
TARGET = 0.90


def horizon():
    p10 = STEP_OK ** H_SHORT
    p100 = STEP_OK ** H_LONG
    eps_req = 1 - TARGET ** (1 / H_LONG)
    ratio = (1 - STEP_OK) / eps_req
    eps_h = -math.log(TARGET)          # small-eps form: eps * H = -ln(0.9)
    # a check after every 10 turns that catches and restores (perfect, as in
    # the notebook's systems insight): each 10-turn segment succeeds 90.4% of
    # the time, so a segment needs 1/0.904 tries on average
    seg_tries = 1 / p10
    p200 = STEP_OK ** (2 * H_LONG)       # twice the turns squares success
    return dict(p200=p200, p10=p10, p100=p100, eps_req=eps_req, ratio=ratio, eps_h=eps_h,
                seg_tries=seg_tries)


# ----------------------------------------------------------------- NM 1.2
# CH Napkin Math 1.2: attempt = $1.30 tokens + $0.80 sandbox, tests, review;
# accepted with probability 0.40. Options: halve the token price; raise
# acceptance to 0.60.
TOKENS, OTHER, P_ACC, P_BETTER, CUT = 1.30, 0.80, 0.40, 0.60, 0.5


def cost():
    attempt = TOKENS + OTHER
    base = attempt / P_ACC
    cheaper = TOKENS * (1 - CUT) + OTHER
    price_cut = cheaper / P_ACC
    better = attempt / P_BETTER
    share = TOKENS / attempt
    breakeven = P_ACC / (1 - CUT * share)
    attempts = 1 / P_ACC
    understate = base / attempt          # CH 1.9 pitfall: charging only the winner
    # CH Checkpoint 1.1, pricing the choice: halve the $0.80; or +10 points
    halve_other = (TOKENS + OTHER / 2) / P_ACC
    plus_ten = attempt / (P_ACC + 0.10)
    # CH 1.9 pitfall: 5 independent attempts at 40 percent
    k5 = 1 - (1 - P_ACC) ** 5
    return dict(attempt=attempt, base=base, cheaper=cheaper, price_cut=price_cut,
                better=better, share=share, breakeven=breakeven, attempts=attempts,
                understate=understate, halve_other=halve_other, plus_ten=plus_ten,
                k5=k5)


# ----------------------------------------------------- CH 1.9, first fallacy
# 30 turns, first prompt 8,000 tokens, 1,500 appended per turn, 128,000 window.
TURNS, FIRST, APPEND, WINDOW = 30, 8000, 1500, 128000


def window():
    last = FIRST + (TURNS - 1) * APPEND
    resent = sum(FIRST + t * APPEND for t in range(TURNS))
    return dict(last=last, used=last / WINDOW, resent=resent, ratio=resent / last)


# ------------------------------------------------------------------- RUN
def run_rows():
    with open(HERE / "data" / "swebench_verified_mini_sonnet4.csv") as f:
        return list(csv.DictReader(f))


CALL_BINS = [(1, 20, "up to 20"), (21, 30, "21 to 30"), (31, 40, "31 to 40"),
             (41, 60, "41 to 60"), (61, 10 ** 6, "over 60")]


def real_run():
    rows = run_rows()
    n = len(rows)
    calls = [int(r["model_calls"]) for r in rows]
    costs = [float(r["cost_usd"]) for r in rows]
    ok = [r for r in rows if r["resolved"] == "1"]
    acc = len(ok)
    total = sum(costs)
    mean_attempt = total / n
    p_acc = acc / n
    per_accepted = mean_attempt / p_acc          # equals total / acc here
    winners_only = statistics.mean(float(r["cost_usd"]) for r in ok)
    submitted = sum(r["exit_status"] == "Submitted" for r in rows)
    cmd = {k: sum(int(r[f"cmd_{k}"]) for r in rows) for k in ("read", "write", "run", "net")}
    cmds = sum(cmd.values())
    obs = sum(int(r["observations"]) for r in rows)
    nonzero = sum(int(r["nonzero_exit"]) for r in rows)
    trunc = sum(int(r["truncated"]) for r in rows)
    syntax = [r for r in rows if r["syntax_error_seen"] == "1"]
    bins = []
    for lo, hi, lab in CALL_BINS:
        b = [r for r in rows if lo <= int(r["model_calls"]) <= hi]
        bins.append((lab, len(b), sum(r["resolved"] == "1" for r in b)))
    return dict(n=n, acc=acc, rejected=n - acc, p_acc=p_acc, total=total,
                mean_attempt=mean_attempt, per_accepted=per_accepted,
                winners_only=winners_only, understate=per_accepted / winners_only,
                submitted=submitted, calls_median=statistics.median(calls),
                calls_min=min(calls), calls_max=max(calls), calls=calls,
                cmd=cmd, cmds=cmds, obs=obs, nonzero=nonzero, trunc=trunc,
                syntax_n=len(syntax), syntax_ok=sum(r["resolved"] == "1" for r in syntax),
                bins=bins, rows=rows)


# Hook task: astropy__astropy-14995 in the same run (RUN, SWEB).
HOOK_ID = "astropy__astropy-14995"
HOOK_P2P = 179          # SWEB: PASS_TO_PASS tests that must keep passing
HOOK_F2P = 1            # SWEB: test_nddata_bitmask_arithmetic


def hook_task():
    r = next(r for r in run_rows() if r["instance_id"] == HOOK_ID)
    return dict(calls=int(r["model_calls"]), cost=float(r["cost_usd"]),
                resolved=r["resolved"] == "1", files=int(r["files_in_submission"]),
                tests=HOOK_P2P + HOOK_F2P)


# ------------------------------------------------------------------ METR
def metr():
    sys.path.insert(0, str(MLSYSIM))
    import numpy as np
    from mlsysim import Literature
    rows = Literature.AgentTimeHorizon.entries
    frontier = [r for r in rows if r["is_sota"]]
    fit = [r for r in frontier if r["p50_min"] <= 16 * 60]       # the source's rule
    recent = [r for r in fit if r["release_date"] >= "2023-01-01"]

    def days(pts):
        t = np.array([dt.date.fromisoformat(r["release_date"]).toordinal() for r in pts])
        return 1 / np.polyfit(t, np.log2([r["p50_min"] for r in pts]), 1)[0]

    first_s = frontier[0]["p50_min"] * 60
    last_h = frontier[-1]["p50_min"] / 60
    return dict(rows=rows, first_s=first_s, last_h=last_h,
                growth=frontier[-1]["p50_min"] / frontier[0]["p50_min"],
                all_months=days(fit) / 30.44, recent_months=days(recent) / 30.44,
                first_date=frontier[0]["release_date"], last_date=frontier[-1]["release_date"])


if __name__ == "__main__":
    h = horizon()
    print("NM 1.1 horizon compounding (CH)")
    show("success at 10 turns", f"{h['p10']:.3f} -> 90.4%")
    show("success at 100 turns", f"{h['p100']:.3f} -> 36.6%")
    show("step error for 90% at 100 turns", f"{h['eps_req']:.5f} -> 0.11% per step")
    show("times lower than 1%", f"{h['ratio']:.1f}x")
    show("eps*H for 90% (any horizon)", f"{h['eps_h']:.3f}")
    show("success at 200 turns", f"{h['p200']:.3f} -> 13.4% = 0.366 squared")
    show("tries per checked 10-turn segment", f"{h['seg_tries']:.2f}")

    c = cost()
    print("NM 1.2 cost per accepted task (CH)")
    show("attempt", f"${c['attempt']:.2f}")
    show("attempts per acceptance", f"{c['attempts']:.1f}")
    show("baseline C_eff", f"${c['base']:.2f}")
    show("cheaper attempt (half token price)", f"${c['cheaper']:.2f}")
    show("C_eff, half token price", f"${c['price_cut']:.3f} -> $3.63")
    show("C_eff, acceptance 0.60", f"${c['better']:.2f}")
    show("token share of attempt", f"{c['share']:.3f} -> 62%")
    show("break-even acceptance", f"{c['breakeven']:.3f} -> 0.58")
    show("winner-only understatement", f"{c['understate']:.1f}x")
    show("Checkpoint: halve the $0.80", f"${c['halve_other']:.2f}")
    show("Checkpoint: acceptance +10 points", f"${c['plus_ten']:.2f}")
    show("5 independent attempts at 40%", f"{c['k5']:.3f} -> 92%")

    w = window()
    print("1.9 window fallacy (CH)")
    show("final call reads", f"{w['last']:,} tokens")
    show("share of 128,000 window", f"{w['used']:.3f} -> 40%")
    show("cumulative input resent", f"{w['resent']:,} tokens")
    show("ratio to final context", f"{w['ratio']:.1f}x")

    r = real_run()
    print("RUN SWE-bench Verified, one attempt per task")
    show("tasks", r["n"])
    show("submitted with exit status Submitted", r["submitted"])
    show("accepted by hidden tests", f"{r['acc']} ({r['p_acc']:.1%})")
    show("rejected", f"{r['rejected']} ({r['rejected'] / r['n']:.1%})")
    show("total spend", f"${r['total']:.2f}")
    show("mean attempt cost", f"${r['mean_attempt']:.3f}")
    show("cost per accepted task", f"${r['per_accepted']:.3f}")
    show("mean cost of accepted attempts only", f"${r['winners_only']:.3f}")
    show("understatement if only winners are charged", f"{r['understate']:.2f}x")
    show("model calls per task: median, min, max",
         f"{r['calls_median']}, {r['calls_min']}, {r['calls_max']}")
    show("commands", f"{r['cmds']:,}")
    for k, v in r["cmd"].items():
        show(f"  {k}", f"{v:,} ({v / r['cmds']:.1%})")
    show("observations", f"{r['obs']:,}")
    show("  nonzero exit", f"{r['nonzero']:,} ({r['nonzero'] / r['obs']:.1%})")
    show("  output too long", f"{r['trunc']:,} ({r['trunc'] / r['obs']:.1%})")
    show("tasks where a syntax error appeared", f"{r['syntax_n']} (accepted {r['syntax_ok']})")
    for lab, nb, ok in r["bins"]:
        show(f"  calls {lab}", f"{ok}/{nb} accepted ({ok / nb:.0%})")

    k = hook_task()
    print(f"HOOK {HOOK_ID} (RUN, SWEB)")
    show("model calls", k["calls"])
    show("cost", f"${k['cost']:.2f}")
    show("resolved", k["resolved"])
    show("files in submitted diff", k["files"])
    show("hidden tests run", k["tests"])

    m = metr()
    print("METR time horizon v1.1 (frontier at release)")
    show("first frontier horizon", f"{m['first_s']:.1f} s ({m['first_date']})")
    show("latest frontier horizon", f"{m['last_h']:.1f} h ({m['last_date']})")
    show("growth", f"{m['growth']:,.0f}x -> about 19,000")
    show("doubling, whole period", f"{m['all_months']:.1f} months")
    show("doubling, since 2023", f"{m['recent_months']:.1f} months")
