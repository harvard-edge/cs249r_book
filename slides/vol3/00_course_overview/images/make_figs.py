#!/usr/bin/env python3
"""Figures for Lecture 0 (Volume III course overview).

Every value comes from calcs.py, which reads MLSysIM:
  - SWE-bench Verified leaderboard submissions
    (Scenarios.SweBenchVerifiedLeaderboardOneAttempt over
    slides/vol3/00_course_overview/images/data/swe_bench_verified_leaderboard.csv)
  - The django__django-12050 run of mini-SWE-agent 1.0.0 with Claude 4 Sonnet
    (vol3/01_introduction/data/swe_bench_verified_mini_v1_sonnet4_django_12050_calls.csv)
This file types only dates, layout, and label offsets.

Run from this folder: python3 make_figs.py   (then make deck converts SVG to PDF)
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import slideplot as sp  # noqa: E402
import calcs  # noqa: E402
from deckvals import num, val  # noqa: E402

plt = sp.plt
C = sp.C
OUT = Path(__file__).parent


def trend_plot():
    """Resolved share of SWE-bench Verified, one-attempt submissions, full width."""
    lb = calcs.LB
    pts = [(dt.date.fromisoformat(r["date"]), 100 * lb.share(r["resolved"])) for r in lb.comparable]
    front = [(dt.date.fromisoformat(d), 100 * s) for d, s in lb.frontier]
    end = dt.date(2026, 3, 1)
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=34, color=C["neutral_l"],
               edgecolors=C["neutral"], linewidths=0.6, zorder=3)
    fx = [p[0] for p in front] + [end]
    fy = [p[1] for p in front] + [front[-1][1]]
    ax.step(fx, fy, where="post", color=C["compute"], lw=2.6, zorder=4)
    first, best = front[0], front[-1]
    ax.annotate(f"Oct 2023: {val('lb_first_pct')}%", first, xytext=(dt.date(2023, 8, 20), 88), textcoords="data",
                fontsize=13, fontweight="bold", color=C["compute"],
                arrowprops=dict(arrowstyle="-", color=C["compute"], lw=1))
    ax.annotate(f"Dec 2025: {val('lb_best_pct')}%", best, xytext=(-150, 14), textcoords="offset points",
                fontsize=13, fontweight="bold", color=C["compute"],
                arrowprops=dict(arrowstyle="-", color=C["compute"], lw=1))
    ax.set_ylim(0, 100)
    ax.set_xlim(dt.date(2023, 8, 1), end)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks([dt.date(y, 1, 1) for y in (2024, 2025, 2026)])
    ax.set_xticklabels(["2024", "2025", "2026"])
    ax.set_ylabel("Issues resolved (%)")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, OUT / "trend.svg")


def loop_bars():
    """Same model, retrieval baseline (no loop) vs agent loop; two steps."""
    pairs = [dict(model=name, one=num(f"{key}_one"), loop=num(f"{key}_loop"))
             for key, name, _, _ in calcs.PAIRS]
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.4, 3.0))
        y = [1.0, 0.0]
        for yi, p in zip(y, pairs):
            ax.barh(yi + 0.2, p["one"], height=0.36, color=C["neutral_l"], edgecolor=C["neutral"])
            ax.text(p["one"] + 2, yi + 0.2, f"{p['one']:.0f}  one call", va="center", fontsize=14, color=C["text"])
            if step == 2:
                ax.barh(yi - 0.2, p["loop"], height=0.36, color=C["compute"])
                ax.text(p["loop"] + 2, yi - 0.2, f"{p['loop']:.0f}  loop", va="center", fontsize=14,
                        fontweight="bold", color=C["compute"])
            else:
                ax.text(4, yi - 0.2, "in a loop: ?", va="center", fontsize=14, color=C["danger"])
        ax.set_yticks(y)
        ax.set_yticklabels([p["model"] for p in pairs], fontsize=14)
        ax.set_xlim(0, 150)
        ax.set_xlabel(f"Issues resolved, of {val('swe_tasks')}", fontsize=13)
        ax.tick_params(axis="x", labelsize=12)
        ax.set_ylim(-0.5, 1.5)
        ax.grid(axis="x", color=C["grid"], lw=0.6)
        ax.set_axisbelow(True)
        sp.save(fig, OUT / f"loop_bars_step{step}.svg")


calls_rows = calcs.CALLS


def context_bars():
    """Characters each model call read in the real django run; column slot."""
    calls = [int(r["call"]) for r in calls_rows]
    chars = [r["context_chars"] / 1e3 for r in calls_rows]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.bar(calls, chars, color=C["memory"], width=0.75)
    ax.annotate(f"call 1: {val('ctx_first_k')}k", (calls[0], chars[0]), xytext=(-8, 26), textcoords="offset points",
                ha="left", fontsize=13, fontweight="bold", color=C["memory"])
    ax.annotate(f"call {val('ctx_last_call')}: {val('ctx_last_k')}k", (calls[-1], chars[-1]), xytext=(4, 6), textcoords="offset points",
                ha="right", fontsize=13, fontweight="bold", color=C["memory"])
    ax.set_xlabel("Model call", fontsize=13)
    ax.set_ylabel("Characters read (thousands)", fontsize=13)
    ax.set_xticks([1, 5, 10, 15, 20, calls[-1]])
    ax.tick_params(labelsize=12)
    ax.set_ylim(0, 35)
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, OUT / "context_bars.svg")


if __name__ == "__main__":
    trend_plot()
    loop_bars()
    context_bars()
    print("figures written")
