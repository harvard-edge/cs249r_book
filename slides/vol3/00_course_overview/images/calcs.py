#!/usr/bin/env python3
"""Every derived number Lecture 0 (Volume III) shows, computed from sourced inputs.

Run from this folder: python3 calcs.py
Each line prints the value with the slide (or note) that shows it. Copy the
printed values onto the slides; never compute a slide number by hand.
make_figs.py imports the functions below so plots and slides agree.

Lecture 0 reveals no Lecture 1 or Lecture 2 result on a slide. The two
first-taste questions (horizon and resend) are computed here only so the
answer choices and the instructor's note are right; the slides do not show
the answers.
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
TASKS = 500  # SWE-bench Verified tasks (Chowdhury et al. 2024)


# --- The hook: one real repair (SWE-bench Verified submission
#     20250726_mini-v1.0.0_claude-sonnet-4-20250514, per_instance_details.json)
def hook_run():
    det = json.loads((HERE / "swebench_per_instance.json").read_text())["django__django-12050"]
    return dict(cost=det["cost"], calls=det["api_calls"], resolved=det["resolved"])


# --- Why now: SWE-bench Verified leaderboard (swebench_trend.py writes the CSV
#     from github.com/SWE-bench/experiments, evaluation/verified/*)
def trend():
    rows = list(csv.DictReader(open(HERE / "swebench_verified_trend.csv")))
    one = [r for r in rows if r["attempts"] == "1"]
    one.sort(key=lambda r: r["date"])
    first = one[0]
    best = max(one, key=lambda r: int(r["resolved"]))
    frontier, top = [], -1
    for r in one:
        v = int(r["resolved"])
        if v > top:
            top = v
            frontier.append((r["date"], v / TASKS))
    return dict(points=[(r["date"], int(r["resolved"]) / TASKS) for r in one],
                n=len(one), n_all=len(rows), first=first, best=best,
                first_share=int(first["resolved"]) / TASKS,
                best_share=int(best["resolved"]) / TASKS, frontier=frontier)


# --- The same model with and without a loop (same leaderboard, 2 April 2024):
#     SWE-bench's retrieval baseline (retrieved files in, one patch out)
#     against SWE-agent's loop, for GPT-4 (1106) and Claude 3 Opus.
PAIRS = [("GPT-4 (1106)", "20240402_rag_gpt4", "20240402_sweagent_gpt4"),
         ("Claude 3 Opus", "20240402_rag_claude3opus", "20240402_sweagent_claude3opus")]


def loop_vs_one_call():
    rows = {r["folder"]: int(r["resolved"]) for r in csv.DictReader(open(HERE / "swebench_verified_trend.csv"))}
    out = []
    for name, rag, agent in PAIRS:
        out.append(dict(model=name, one=rows[rag], loop=rows[agent], ratio=rows[agent] / rows[rag],
                        one_share=rows[rag] / TASKS, loop_share=rows[agent] / TASKS))
    return out


# --- First taste 1: horizon (Introduction, Napkin Math 1.1; Lecture 1 owns it)
STEP_OK, H_LONG = 0.99, 100


def horizon_answer():
    return STEP_OK ** H_LONG


# --- First taste 2: resend (The Foundation Model, Napkin Math 2.2; Lecture 2 owns it)
TURNS, FIRST_PROMPT, APPEND = 30, 8_000, 1_500


def resend_answer():
    return TURNS * FIRST_PROMPT + APPEND * TURNS * (TURNS - 1) // 2


if __name__ == "__main__":
    h = hook_run()
    print("[Hook / predict cost] django__django-12050: calls", h["calls"], " cost $", f"{h['cost']:.2f}",
          " resolved", h["resolved"])
    t = trend()
    print("[Trend] one-attempt submissions plotted", t["n"], "of", t["n_all"], "with results files")
    print("[Trend] first", t["first"]["date"], t["first"]["name"], t["first"]["resolved"], "of", TASKS,
          f"= {t['first_share']:.1%}")
    print("[Trend] best", t["best"]["date"], t["best"]["name"], t["best"]["resolved"], "of", TASKS,
          f"= {t['best_share']:.1%}")
    for p in loop_vs_one_call():
        print(f"[Loop bars] {p['model']}: one call {p['one']} ({p['one_share']:.1%}), loop {p['loop']} "
              f"({p['loop_share']:.1%}), ratio {p['ratio']:.1f}x")
    print("[First taste, notes only] 0.99^100 =", f"{horizon_answer():.1%}")
    print("[First taste, notes only] 30-turn resent input tokens =", f"{resend_answer():,}")
