#!/usr/bin/env python3
"""Every number on the Volume III Lecture 20 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (registry, Literature, series) and the calculations
(Scenarios.*); the chapter reads the same scenarios. A number this lecture
recalls from an earlier one is read from the scenario or Literature entry that
lecture's deck reads, so the recall is the same object, not a copy. This file
only selects values, converts units, and formats them; it types no numbers.
Each value is recorded with v() under the key the deck prints as \\val{key};
running this file writes images/numbers.tex.

Sources
  Incident bill       Scenarios.RemediationIncidentAttempt (Agents.Remediation.OnCall),
                      Scenarios.RemediationCanaryPivotLoss (expected irreversible loss)
  Lost reply          Scenarios.BlindRetryAgainstStatusLookup (hypothetical rates and prices)
  Measured run        data/settle_run.json: the deck's localhost run (data/settle_run.py)
  Passing set         Scenarios.ThirtyPercentValidPassingSet
  SWE-bench+          Scenarios.SweAgentGpt4PassingPatchAudit (Literature.AgentEvaluation)
  SWE-agent           Literature.AgentRetrieval (Yang et al. 2024, Tables 1 and 3)
  Release evidence    Scenarios.AcceptanceRatePlanning (Lecture 13's detectable differences)
  Fallacy             Scenarios.ResidualViolationsFiftyTurns
  Health window       Scenarios.HalvedHealthWindow
  Attribution         Scenarios.CheckedWindowBlameSearch
  Specification       Scenarios.SpecificationReviewHalfOfEffort (Amdahl)
  Two relations       Scenarios.UncheckedRepairHorizon, Scenarios.RepairAttemptCostLevers (Lecture 1)
  METR p50 and p80    Scenarios.FrontierHorizonReliabilityGap (Literature.AgentTimeHorizon)
  Recalled lectures   L2 Scenarios.CodingAgentResend30Turns; L4 Scenarios.RepairContextBudget;
                      L7 Scenarios.BlindRetryTwoPercentLoss; L10 Scenarios.FreezeRuleOverStepBudget;
                      L17 Literature.AgentEvaluation (METR o3); L18 Scenarios.ResearchSystemTokenBreakEven
  Not yet wired       the Lecture 19 recall (100 tickets, $1.32 against $0.55): its Agent
                      Economics scenario is being added with the Lecture 19 deck.

Run from this folder: python3 calcs.py
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Agents, Scenarios, Literature = mlsysim.Agents, mlsysim.Scenarios, mlsysim.Literature
HERE = Path(__file__).resolve().parent

REMED = Agents.Remediation.OnCall
BILL = Scenarios.RemediationIncidentAttempt
PIVOT = Scenarios.RemediationCanaryPivotLoss
LOST = Scenarios.BlindRetryAgainstStatusLookup
PASSING = Scenarios.ThirtyPercentValidPassingSet
AUDIT = Scenarios.SweAgentGpt4PassingPatchAudit
SWE = Literature.AgentRetrieval
PLAN = Scenarios.AcceptanceRatePlanning
RESIDUAL = Scenarios.ResidualViolationsFiftyTurns
WINDOW = Scenarios.HalvedHealthWindow
BLAME = Scenarios.CheckedWindowBlameSearch
SPEC = Scenarios.SpecificationReviewHalfOfEffort
HORIZON = Scenarios.UncheckedRepairHorizon
COST = Scenarios.RepairAttemptCostLevers
GAP = Scenarios.FrontierHorizonReliabilityGap
RESEND = Scenarios.CodingAgentResend30Turns
BUDGET = Scenarios.RepairContextBudget
RETRY = Scenarios.BlindRetryTwoPercentLoss
RULE = Scenarios.FreezeRuleOverStepBudget
METR = Literature.AgentEvaluation
TOKENS = Scenarios.ResearchSystemTokenBreakEven


def pct(x):
    return 100 * x


def usd(q):
    return q.to("USD").magnitude


def measured_run():
    """data/settle_run.json: the deck's localhost run of the four runtimes."""
    return json.loads((HERE / "data" / "settle_run.json").read_text())


print("== The lost reply (Scenarios.BlindRetryAgainstStatusLookup)")
lr = LOST.result
v("lr_calls", LOST.calls, "{:,.0f}", "mutating calls a day")
v("lr_req", LOST.request_loss, "{:.3f}", "request lost")
v("lr_rep", LOST.reply_loss, "{:.3f}", "reply lost")
v("lr_keep", 1 - LOST.request_loss, "{:.3f}", "request delivered")
v("lr_dup_cost", usd(LOST.duplicate_cost), "{:,.0f}", "a duplicated change, $")
v("lr_lookup", usd(LOST.status_lookup_cost), "{:.2f}", "a status lookup, $")
v("lr_timeouts", lr.timeouts, "{:.1f}", "timeouts a day")
v("lr_dups", lr.duplicates, "{:.1f}", "blind-retry duplicates a day")
v("lr_dup_day", usd(LOST.duplicate_cost_per_day), "{:,.0f}", "duplicates, $ a day")
v("lr_lookup_day", usd(LOST.lookup_cost_per_day), "{:.2f}", "lookups on every timeout, $ a day")
v("lr_false", LOST.unapplied_closures, "{:.0f}", "assume success: expected closures with nothing applied")

print("== Measured run (data/settle_run.json)")
run = measured_run()
assert (run["p_request_lost"], run["p_reply_lost"]) == (LOST.request_loss, LOST.reply_loss), \
    "the run used the scenario's loss rates"
rt = run["runtimes"]
n_runs = {r["incidents"] for r in rt.values()}
assert n_runs == {LOST.calls}, "every runtime handled one day's calls"
v("ms_n", n_runs.pop(), "{:,.0f}", "incidents per runtime")
v("ms_dups", rt["retry"]["duplicates"], "{:.0f}", "retry blindly: duplicated changes")
v("ms_false", rt["assume"]["false_closures"], "{:.0f}", "assume success: closed, never applied")
v("ms_halt", rt["halt"]["unowned_unknowns"], "{:.0f}", "halt: no recorded outcome")
v("ms_halt_applied", rt["halt"]["applied_but_unknown"], "{:.0f}", "halt: of them, already applied")
v("ms_settle_words", sum(rt["settle"][k] for k in ("duplicates", "false_closures", "unowned_unknowns")),
  "words", "settle by status: failures, in words")

print("== The incident's bill (Scenarios.RemediationIncidentAttempt, RemediationCanaryPivotLoss)")
b = BILL.result
v("bill_model", usd(REMED.model_cost_per_attempt), "{:.2f}", "model calls, $")
v("bill_tools", usd(REMED.environment_cost_per_attempt), "{:.2f}", "telemetry and change tools, $")
v("bill_health", usd(REMED.assessment_cost_per_attempt), "{:.2f}", "health-window check, $")
v("bill_approval", usd(REMED.approval_cost_per_attempt), "{:.0f}", "on-call review, $")
v("bill_approval2", usd(REMED.approval_cost_per_attempt), "{:.2f}", "on-call review, $ (figure label)")
v("bill_attempt", b.attempt_cost, "{:.2f}", "attempt, $")
v("bill_acc_pct", pct(BILL.acceptance), "{:.0f}", "acceptance, %")
v("bill_acc", BILL.acceptance, "{:.2f}", "acceptance, fraction")
v("bill_cpa", b.cost_per_accepted, "{:.2f}", "per accepted incident, $")
v("bill_model_share", pct(b.token_share), "{:.1f}", "model share of the attempt, %")
v("bill_approval_share", pct(REMED.approval_share), "{:.1f}", "approval share of the attempt, %")
v("bill_p_wrong", pct(REMED.wrong_change_probability), "{:.0f}", "wrong change proposed, % of attempts")
v("bill_impact", usd(REMED.wrong_change_impact), "{:,.0f}", "a committed wrong change, $")
v("bill_loss", usd(PIVOT.result.loss_without_check), "{:.0f}", "expected irreversible loss, no check, $")
v("bill_loss_mult", usd(PIVOT.result.loss_without_check) / b.attempt_cost, "{:.1f}", "the loss over the attempt")
v("bill_attempt2", b.grown_attempt_cost, "{:.2f}", "attempt with twice the model calls, $")
v("bill_rise", pct(b.grown_attempt_rise), "{:.2f}", "attempt rise, %")
v("bill_acc_needed", pct(b.grown_break_even_acceptance), "{:.2f}", "acceptance needed, %")

print("== The passing set (Scenarios.ThirtyPercentValidPassingSet)")
ps = PASSING.result
v("ps_valid", pct(1 - PASSING.defect_rate), "{:.0f}", "valid candidates, %")
v("ps_fr", pct(1 - PASSING.valid_accept_rate), "{:.0f}", "valid ones the check misses, %")
v("ps_fa", pct(PASSING.false_accept_rate), "{:.0f}", "invalid ones the check passes, %")
v("ps_invalid", pct(PASSING.defect_rate), "{:.0f}", "invalid candidates, %")
v("ps_valid_frac", 1 - PASSING.defect_rate, "{:.2f}", "valid, fraction")
v("ps_keep_frac", PASSING.valid_accept_rate, "{:.2f}", "valid passed, fraction")
v("ps_invalid_frac", PASSING.defect_rate, "{:.2f}", "invalid, fraction")
v("ps_fa_frac", PASSING.false_accept_rate, "{:.2f}", "invalid passed, fraction")
v("ps_pass_valid", ps.pass_share - ps.defective_pass, "{:.2f}", "valid and passing, of all candidates")
v("ps_pass_invalid", ps.defective_pass, "{:.3f}", "invalid and passing, of all candidates")
v("ps_pass", ps.pass_share, "{:.3f}", "passing, of all candidates")
v("ps_after", pct(ps.posterior), "{:.1f}", "invalid share of the passing set, %")
v("ps_fa_ok", pct(1 - PASSING.false_accept_rate), "{:.0f}", "the check is right on invalid ones, %")

print("== SWE-bench+ (Scenarios.SweAgentGpt4PassingPatchAudit)")
v("sbp_total", AUDIT.total, "{:.0f}", "passing patches reviewed")
v("sbp_leak", AUDIT.solution_leak, "{:.0f}", "solution in the issue")
v("sbp_weak", AUDIT.weak_tests, "{:.0f}", "passed weak tests")
v("sbp_incorrect", AUDIT.incorrect_fix, "{:.0f}", "incorrect")
v("sbp_incomplete", AUDIT.incomplete_fix, "{:.0f}", "incomplete")
v("sbp_other", AUDIT.different_files, "{:.0f}", "changed other files")
v("sbp_correct", AUDIT.correct, "{:.0f}", "correct")
v("sbp_diff", AUDIT.different_correct, "{:.0f}", "correct by a different route")
v("sbp_suspect", AUDIT.suspicious, "{:.0f}", "suspect")
v("sbp_suspect_pct", pct(AUDIT.share(AUDIT.suspicious)), "{:.2f}", "suspect, %")

print("== SWE-agent (Literature.AgentRetrieval)")
v("swa_agent", SWE.SweBenchLiteSweAgentResolved, "{:.1f}", "SWE-agent interface, %")
v("swa_noedit", SWE.SweAgentNoEditCommandResolved, "{:.1f}", "no edit command, %")
v("swa_n", SWE.SweBenchLiteInstances, "{:.0f}", "SWE-bench Lite issues")

print("== Release evidence (Scenarios.AcceptanceRatePlanning: eq. 13.3, n proportional to 1/delta^2)")
wide, narrow = PLAN.differences
v("rel_halve", (wide / narrow) ** 2, "{:.0f}", "tasks per arm when the difference halves, x")
v("rel_shrink", math.sqrt(narrow / wide), "{:.2f}", "difference after doubling the tasks, of before")

print("== The capable-model fallacy (Scenarios.ResidualViolationsFiftyTurns)")
v("fc_eps", pct(RESIDUAL.step_violation), "{:.1f}", "residual violations per step, %")
v("fc_ok", 1 - RESIDUAL.step_violation, "{:.3f}", "per-step compliance")
v("fc_h", RESIDUAL.horizon, "{:.0f}", "turns")
v("fc_pct", pct(RESIDUAL.trajectory_violation), "{:.2f}", "trajectories with a violation, %")
v("fc_day", RESIDUAL.trajectories_per_day, "{:,.0f}", "trajectories a day")
v("fc_per_day", RESIDUAL.violating_per_day, "{:.0f}", "hazardous proposals a day")

print("== Shortened health window (Scenarios.HalvedHealthWindow)")
v("hw_widen", WINDOW.interval_widening, "{:.2f}", "interval widens, x")

print("== Attribution (Scenarios.CheckedWindowBlameSearch)")
v("at_b", BLAME.branching, "{:.0f}", "candidate actions per step")
v("at_b_words", BLAME.branching, "words", "the same, in words")
v("at_h", BLAME.horizon, "{:.0f}", "steps")
v("at_exp", round(BLAME.full_exponent), "{:.0f}", "full search, power of ten")
v("at_w", BLAME.window, "{:.0f}", "steps between checks")
v("at_w_words", BLAME.window, "words", "the same, in words")
v("at_window", BLAME.window_paths, "{:,.0f}", "paths in one window")
v("at_sec", BLAME.replay_time.to("s").magnitude, "{:.1f}", "per replayed check, s")
v("at_min", BLAME.window_search_time.to("minute").magnitude, "{:.0f}", "window search, minutes")

print("== Specification boundary (Scenarios.SpecificationReviewHalfOfEffort)")
sp = SPEC.result
v("sp_serial", 1 - sp.stage_fraction, "{:.1f}", "serial fraction")
v("sp_cap", sp.overall_speedup, "{:.0f}", "speedup cap, x")

print("== Lecture 1: horizon compounding (Scenarios.UncheckedRepairHorizon)")
h = HORIZON.result
v("hz_step", pct(HORIZON.step_success), "{:.0f}", "per-step success, %")
v("hz_short", HORIZON.short_horizon, "{:.0f}", "short horizon, turns")
v("hz_long", HORIZON.long_horizon, "{:.0f}", "long horizon, turns")
v("hz_p_short", pct(h.success[HORIZON.short_horizon]), "{:.1f}", "success at the short horizon, %")
v("hz_p_long", pct(h.success[HORIZON.long_horizon]), "{:.1f}", "success at the long horizon, %")

print("== Lecture 1: cost per accepted task (Scenarios.RepairAttemptCostLevers)")
c = COST.result
v("c_attempt", c.attempt_cost, "{:.2f}", "attempt, $")
v("c_token_share", pct(c.token_share), "{:.0f}", "token share, %")
v("c_acc", pct(COST.acceptance), "{:.0f}", "acceptance, %")
v("c_better_acc", pct(COST.improved_acceptance), "{:.0f}", "improved acceptance, %")
v("c_eff", c.cost_per_accepted, "{:.2f}", "per accepted task, $")
v("c_cheap", c.cost_per_accepted_cheaper_tokens, "{:.2f}", "half the token price, $")
v("c_better", c.cost_per_accepted_improved, "{:.2f}", "better acceptance, $")

print("== METR p50 against p80 (Scenarios.FrontierHorizonReliabilityGap)")
v("mt_n", len(GAP.rows), "{:.0f}", "frontier models since 2024")
v("mt_median", GAP.ratio_median, "{:.1f}", "median p50 / p80")
v("mt_min", GAP.ratio_min, "{:.0f}", "smallest ratio")
v("mt_max", GAP.ratio_max, "{:.0f}", "largest ratio")

print("== Recalled lectures")
v("rs_resent", RESEND.result.resent_tokens, "{:,.0f}", "L2: 30 turns resend, input tokens")
v("rs_fresh", RESEND.result.reused_prefix_tokens, "{:,.0f}", "L2: fresh with prefix reuse, tokens")
v("rs_turns", RESEND.turns, "{:.0f}", "L2: turns")
v("b_append_bill", BUDGET.result.append_bill, "{:,.0f}", "L4: appending, input tokens")
v("b_select_bill", BUDGET.result.select_bill, "{:,.0f}", "L4: selecting, input tokens")
v("rt_dup_pct", pct(RETRY.result.duplicate_probability), "{:.2f}", "L7: duplicates, % of calls")
v("rt_dup_pct4", pct(RETRY.result.duplicate_probability), "{:.4f}", "L7: the same, four places")
v("rt_dups", RETRY.result.duplicates, "{:,.0f}", "L7: duplicates a day")
v("rt_calls", RETRY.calls, "{:,.0f}", "L7: mutating calls a day")
hold = RULE.result.success[RULE.short_horizon]
v("fr_step", pct(RULE.step_success), "{:.0f}", "L10: per-turn compliance, %")
v("fr_turns", RULE.short_horizon, "{:.0f}", "L10: turns")
v("fr_hold", pct(hold), "{:.0f}", "L10: rule held on every turn, %")
v("metr_rebench_hacks", METR.MetrO3ReBenchHackRuns, "{:.0f}", "L17: RE-Bench runs that hacked")
v("metr_rebench_runs", METR.MetrO3ReBenchRuns, "{:.0f}", "L17: RE-Bench runs")
v("metr_rebench_pct", pct(METR.MetrO3ReBenchHackRuns / METR.MetrO3ReBenchRuns), "{:.0f}", "L17: RE-Bench, %")
v("metr_hcast_hacks", METR.MetrO3HcastHackRuns, "{:.0f}", "L17: HCAST runs that hacked")
v("metr_hcast_runs", METR.MetrO3HcastRuns, "{:,.0f}", "L17: HCAST runs")
v("metr_hcast_pct", pct(METR.MetrO3HcastHackRuns / METR.MetrO3HcastRuns), "{:.1f}", "L17: HCAST, %")
v("hook_agent", TOKENS.agent_tokens_vs_chat, "{:.0f}", "L18: single agent, tokens over chat")
v("hook_multi", TOKENS.multi_agent_tokens_vs_chat, "{:.0f}", "L18: multi-agent, tokens over chat")

if __name__ == "__main__":
    write(__file__)
