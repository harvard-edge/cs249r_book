#!/usr/bin/env python3
"""Every number on the Volume III Lecture 1 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (registry, Literature series) and the calculations
(Scenarios.*, each backed by a solver in engine/solvers/agents.py); the
chapter reads the same scenarios. This file only selects values, converts
units, and formats them; it types no numbers. Each value is recorded with v()
under the key the deck prints as \\val{key}; running this file writes
images/numbers.tex.

Sources behind the scenarios
  Napkin Math 1.1   Scenarios.UncheckedRepairHorizon (illustrative 99% per step)
  Napkin Math 1.2   Scenarios.RepairAttemptCostLevers (illustrative prices)
  Section 1.9       Scenarios.AppendOnlyRepairContext (SWE-bench harness budget)
  Real run          Scenarios.MiniSweAgentSonnet4Run: SWE-bench Verified, mini-swe-agent
                    1.0.0 with Claude 4 Sonnet, one attempt per task
                    (Literature.SweBenchMiniSonnet4Run, reduced by extract_run.py)
  Hook task         Scenarios.AstropyBitmaskRepair (astropy__astropy-14995)
  METR              Scenarios.FrontierTaskHorizonTrend (Literature.AgentTimeHorizon)

Run from this folder: python3 calcs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Agents, Scenarios = mlsysim.Agents, mlsysim.Scenarios
Q_ = mlsysim.core.units.Q_

HORIZON = Scenarios.UncheckedRepairHorizon
COST = Scenarios.RepairAttemptCostLevers
CONTEXT = Scenarios.AppendOnlyRepairContext
RUN = Scenarios.MiniSweAgentSonnet4Run
HOOK = Scenarios.AstropyBitmaskRepair
TREND = Scenarios.FrontierTaskHorizonTrend
HARNESS = Agents.Coding.SWE_Bench_Runner


def pct(x):
    return 100 * x


print("== Hook task (astropy__astropy-14995)")
v("hook_calls", HOOK.model_calls, "{:.0f}", "model calls (turns)")
v("hook_cost", HOOK.cost, "{:.2f}", "cost, $")
v("hook_f2p", HOOK.fail_to_pass_tests, "{:.0f}", "new hidden test")
v("hook_p2p", HOOK.pass_to_pass_tests, "{:.0f}", "existing tests that must keep passing")

print("== Real run, 500 tasks, one attempt each")
r = RUN.result
v("run_tasks", r.tasks, "{:.0f}", "tasks")
v("run_submitted", r.submitted, "{:.0f}", "loops ended by the agent's submission")
v("run_acc", r.accepted, "{:.0f}", "accepted by hidden tests")
v("run_rej", r.rejected, "{:.0f}", "rejected")
v("run_acc_pct", pct(r.acceptance), "{:.1f}", "acceptance, %")
v("run_acc_frac", r.acceptance, "{:.3f}", "acceptance, fraction")
v("run_rej_pct", pct(r.rejected / r.tasks), "{:.0f}", "rejected, %")
v("run_total", r.total_cost, "{:.2f}", "total spend, $")
v("run_mean", r.mean_attempt_cost, "{:.3f}", "mean attempt, $")
v("run_per_acc", r.cost_per_accepted, "{:.2f}", "per accepted task, $")
v("run_per_acc3", r.cost_per_accepted, "{:.3f}", "per accepted task, $ (three places)")
v("run_winners", r.accepted_mean_cost, "{:.3f}", "mean of accepted attempts only, $")
v("run_understate", r.accepted_only_understatement, "{:.1f}", "understatement, winners only")
v("run_understate2", r.accepted_only_understatement, "{:.2f}", "the same, two places")
v("run_calls_min", r.calls_min, "{:.0f}", "fewest model calls")
v("run_calls_med", r.calls_median, "{:.0f}", "median model calls")
v("run_calls_max", r.calls_max, "{:.0f}", "most model calls")
v("run_cmds", r.command_total, "{:,.0f}", "commands")
v("run_write_pct", pct(r.commands["write"] / r.command_total), "{:.0f}", "write commands, %")
v("run_run_pct", pct(r.commands["run"] / r.command_total), "{:.0f}", "run commands, %")
v("run_net_word", r.commands["net"], "words", "network commands, in prose")
v("run_trunc", r.truncated_outputs, "{:.0f}", "outputs truncated with a warning")

print("== Repair harness (registry)")
v("harness_steps", HARNESS.max_trajectory_steps, "{:.0f}", "step budget")
v("harness_window", HARNESS.context_window.to("token").magnitude, "{:,.0f}", "context window, tokens")
v("harness_window_k", HARNESS.context_window.to("token").magnitude / 10 ** 3, "{:.0f}", "the same, thousands (128K)")

print("== Napkin Math 1.1: horizon compounding")
h = HORIZON.result
H_SHORT, H_LONG, H_EXT = HORIZON.short_horizon, HORIZON.long_horizon, HORIZON.extended_horizon
v("step_ok", HORIZON.step_success, "{:.2f}", "per-step success")
v("step_ok_pct", pct(HORIZON.step_success), "{:.0f}", "the same, %")
v("step_err_pct", pct(h.step_error), "{:.0f}", "per-step error, %")
v("h_short", H_SHORT, "{:.0f}", "short horizon, turns")
v("h_long", H_LONG, "{:.0f}", "long horizon, turns")
v("h_ext", H_EXT, "{:.0f}", "doubled horizon, turns")
v("p_short_pct", pct(h.success[H_SHORT]), "{:.1f}", "success at the short horizon, %")
v("p_long_pct", pct(h.success[H_LONG]), "{:.1f}", "success at the long horizon, %")
v("p_long", h.success[H_LONG], "{:.3f}", "the same, fraction")
v("p_ext_pct", pct(h.success[H_EXT]), "{:.1f}", "success at the doubled horizon, %")
v("target", HORIZON.target_success, "{:.1f}", "target success, fraction")
v("target_pct", pct(HORIZON.target_success), "{:.0f}", "target success, %")
v("eps_req", h.required_step_error, "{:.5f}", "required step error")
v("eps_req_pct", pct(h.required_step_error), "{:.2f}", "the same, %")
v("eps_ratio", h.error_ratio, "{:.1f}", "how many times lower")
v("eps_h", h.required_error_horizon, "{:.3f}", "eps * H for the target")

print("== Napkin Math 1.2: cost per accepted task")
c = COST.result
v("c_tokens", COST.model_cost.to("USD").magnitude, "{:.2f}", "tokens per attempt, $")
v("c_other", COST.other_cost.to("USD").magnitude, "{:.2f}", "sandbox, tests, review per attempt, $")
v("p_acc", COST.acceptance, "{:.2f}", "acceptance")
v("p_acc_pct", pct(COST.acceptance), "{:.0f}", "acceptance, %")
v("p_better", COST.improved_acceptance, "{:.2f}", "improved acceptance")
v("p_better_pct", pct(COST.improved_acceptance), "{:.0f}", "improved acceptance, %")
v("token_cut", COST.token_price_cut, "{:.1f}", "token price cut")
v("c_attempt", c.attempt_cost, "{:.2f}", "attempt, $")
v("attempts", c.attempts_per_acceptance, "{:.1f}", "attempts per acceptance")
v("c_eff", c.cost_per_accepted, "{:.2f}", "baseline cost per accepted task, $")
v("c_cheap", c.cheaper_attempt_cost, "{:.2f}", "attempt at half the token price, $")
v("c_eff_cheap", c.cost_per_accepted_cheaper_tokens, "{:.2f}", "cost per accepted, half token price, $")
v("c_eff_better", c.cost_per_accepted_improved, "{:.2f}", "cost per accepted, better acceptance, $")
v("token_share", c.token_share, "{:.2f}", "token share of the attempt")
v("token_share_pct", pct(c.token_share), "{:.0f}", "the same, %")
v("p_breakeven", c.break_even_acceptance, "{:.2f}", "break-even acceptance")
v("c_understate", c.accepted_only_understatement, "{:.1f}", "winner-only understatement")

print("== Section 1.9: append-only context")
x = CONTEXT.result
v("ctx_turns", CONTEXT.turns, "{:.0f}", "turns")
v("ctx_s0", CONTEXT.initial_tokens.to("token").magnitude, "{:,.0f}", "first prompt, tokens")
v("ctx_delta", CONTEXT.tokens_per_turn.to("token").magnitude, "{:,.0f}", "appended per turn, tokens")
v("ctx_last", x.last_call_tokens.to("token").magnitude, "{:,.0f}", "last call reads, tokens")
v("ctx_window_pct", pct(x.window_fraction), "{:.0f}", "share of the window, %")
v("ctx_total", x.cumulative_input_tokens.to("token").magnitude, "{:,.0f}", "input sent in all, tokens")
v("ctx_ratio", x.reread_ratio, "{:.1f}", "total over the last call")

print("== METR time horizon (frontier at release)")
t = TREND.result
v("metr_first_s", Q_(t.first_frontier_minutes, "minute").to("second").magnitude, "{:.0f}", "first frontier, s")
v("metr_last_h", Q_(t.last_frontier_minutes, "minute").to("hour").magnitude, "{:.0f}", "latest frontier, h")
v("metr_growth", round(t.growth, -3), "{:,.0f}", "growth, to the thousand")
v("metr_all_months", t.doubling_all.to("month").magnitude, "{:.1f}", "doubling, whole period, months")
v("metr_recent_months", t.doubling_recent.to("month").magnitude, "{:.1f}", "doubling since 2023, months")

if __name__ == "__main__":
    write(__file__)
