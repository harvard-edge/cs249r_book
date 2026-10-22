#!/usr/bin/env python3
"""Every number on the Volume III Lecture 0 slides and notes, read from MLSysIM.

Lecture 0 previews the course, so each number it shares with a later lecture is
read from the same Literature entry or scenario that lecture's deck reads: the
repair run (Lecture 1, Scenarios.MiniSweAgentSonnet4Run's series), the horizon
question (Lecture 1, Scenarios.UncheckedRepairHorizon), the resend question
(Lecture 2, Scenarios.CodingAgentResend30Turns), and the six incident cards
(Lectures 2, 6, 17, and 18). Lecture 0 reveals no Lecture 1 or 2 answer on a
slide; the notes carry them for the instructor.

Sources behind the values
  Hook run          vol3/01_introduction/data/swe_bench_verified_mini_v1_sonnet4_run.csv (django__django-12050)
  Its calls         vol3/01_introduction/data/swe_bench_verified_mini_v1_sonnet4_django_12050_calls.csv
  Leaderboard       Scenarios.SweBenchVerifiedLeaderboardOneAttempt
                    (slides/vol3/00_course_overview/images/data/swe_bench_verified_leaderboard.csv)
  Incident cards    Literature.ServingIncidents, IncidentReports, AgentEvaluation,
                    MultiAgentSystems; Scenarios.ResearchSystemTokenBreakEven

This file only selects values, converts units, and formats them; it types no
numbers. Run from this folder: python3 calcs.py (writes images/numbers.tex).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402

Literature, Scenarios = mlsysim.Literature, mlsysim.Scenarios
LB = Scenarios.SweBenchVerifiedLeaderboardOneAttempt
HORIZON, RESEND = Scenarios.UncheckedRepairHorizon, Scenarios.CodingAgentResend30Turns
BE = Scenarios.ResearchSystemTokenBreakEven
HOOK_ID = "django__django-12050"
HOOK = load_series("vol3/01_introduction/data/swe_bench_verified_mini_v1_sonnet4_run.csv").where(instance_id=HOOK_ID)[0]
CALLS = load_series("vol3/01_introduction/data/swe_bench_verified_mini_v1_sonnet4_django_12050_calls.csv").entries
# Same model, same day (2 April 2024): retrieval baseline (one call) and SWE-agent (loop)
PAIRS = [("gpt4", "GPT-4 (1106)", "20240402_rag_gpt4", "20240402_sweagent_gpt4"),
         ("opus", "Claude 3 Opus", "20240402_rag_claude3opus", "20240402_sweagent_claude3opus")]


def pct(x):
    return 100 * x


print("== Hook: one real repair (django__django-12050)")
v("hook_calls", HOOK["model_calls"], "{:.0f}", "model calls")
v("hook_cost", HOOK["cost_usd"], "{:.2f}", "cost of the model calls, $")

print("== Leaderboard (one attempt per task)")
first, best = LB.first, LB.best
v("swe_tasks", LB.tasks, "{:.0f}", "SWE-bench Verified tasks")
v("lb_n", len(LB.comparable), "{:.0f}", "one-attempt submissions plotted")
v("lb_n_all", len(LB.submissions), "{:.0f}", "submissions with results files")
v("lb_first", first["resolved"], "{:.0f}", "best in October 2023, resolved")
v("lb_first_pct", pct(LB.share(first["resolved"])), "{:.1f}", "the same, %")
v("lb_best", best["resolved"], "{:.0f}", "best by December 2025, resolved")
v("lb_best_pct", pct(LB.share(best["resolved"])), "{:.1f}", "the same, %")

print("== Loop against one call, same model")
for key, _, one, loop in PAIRS:
    v(f"{key}_one", LB.resolved(one), "{:.0f}", "retrieval baseline, resolved")
    v(f"{key}_loop", LB.resolved(loop), "{:.0f}", "SWE-agent loop, resolved")
    v(f"{key}_ratio", LB.ratio(loop, one), "{:.1f}", "loop over one call, x")

print("== Incident cards")
v("pm_worst_pct", pct(Literature.ServingIncidents.Anthropic2025MisroutedShareWorstHour), "{:.0f}",
  "requests corrupted at the worst hour, %")
v("mof_damages", Literature.IncidentReports.MoffattDamagesCad, "{:.2f}", "damages, CA$")
v("metr_hacks", Literature.AgentEvaluation.MetrO3ReBenchHackRuns, "{:.0f}", "RE-Bench runs that hacked")
v("metr_runs", Literature.AgentEvaluation.MetrO3ReBenchRuns, "{:.0f}", "RE-Bench runs")
v("ra_subagents", Literature.MultiAgentSystems.ResearchSystemEarlySubagents, "{:.0f}", "subagents for simple queries")
v("multi_chat_mult", BE.multi_agent_tokens_vs_chat, "{:.0f}", "multi-agent tokens over chat, x")

print("== First taste 1: horizon (notes only)")
h = HORIZON.result
v("hz_step", HORIZON.step_success, "{:.2f}", "per-step success")
v("hz_step_pct", pct(HORIZON.step_success), "{:.0f}", "the same, %")
v("hz_long", HORIZON.long_horizon, "{:.0f}", "steps")
v("hz_long_pct", pct(h.success[HORIZON.long_horizon]), "{:.1f}", "task success at that horizon, %")
v("hz_long_pct0", pct(h.success[HORIZON.long_horizon]), "{:.0f}", "the same, whole percent (answer choice)")

print("== First taste 2: resend (notes only)")
v("rs_turns", RESEND.turns, "{:.0f}", "turns")
v("rs_s0", RESEND.first_prompt, "{:,.0f}", "first prompt, tokens")
v("rs_delta", RESEND.appended, "{:,.0f}", "appended per turn, tokens")
v("rs_resent", RESEND.result.resent_tokens, "{:,.0f}", "input tokens sent over the run")

print("== The hook run's context, per call (characters)")
v("ctx_first_k", CALLS[0]["context_chars"] / 10 ** 3, "{:.1f}", "call 1 context, thousand characters")
v("ctx_last_k", CALLS[-1]["context_chars"] / 10 ** 3, "{:.1f}", "last call context, thousand characters")
v("ctx_last_call", CALLS[-1]["call"], "{:.0f}", "last call")

if __name__ == "__main__":
    write(__file__)
