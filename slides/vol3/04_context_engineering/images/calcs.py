#!/usr/bin/env python3
"""Every number on the Volume III Lecture 4 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (the coding harness's window and step budget, the
scenario sizes of the repair request, Literature results) and the
calculations (Scenarios.*, each backed by a solver in
engine/solvers/context.py, or the Lecture 1 and 2 scenarios the recall slide
repeats); the chapter reads the same scenarios. The one other source is the
deck's own tokenizer run, artifacts/tokenize_output.txt, read as recorded.
This file only selects values, converts units, and formats them; it types no
numbers. Each value is recorded with v() under the key the deck prints as
\\val{key}; running this file writes images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Literature, Scenarios = mlsysim.Literature, mlsysim.Scenarios
HERE = Path(__file__).resolve().parent

BUDGET = Scenarios.RepairContextBudget
HEAD, LOOK = Scenarios.ToyHeadAttentionDilution, Scenarios.ToyHeadLookalikeExcerpt
SWEEP = Scenarios.FictionLiveBenchThrough60K
LAYOUT, LAYOUT4 = Scenarios.RepairPrefixLayoutBill, Scenarios.RepairPrefixLayoutEveryFourthCall
MARKS = Scenarios.RepairCompactionWatermarks
INVAL = Scenarios.SupersededExcerptInvalidation
RETAIN, CHECK = Scenarios.CompactionConstraintRetention, Scenarios.ProtectedFieldCheckPayback
RESEND, COST = Scenarios.CodingAgentResend30Turns, Scenarios.RepairAttemptCostLevers
USE, SUMM, CACHE = Literature.ContextUse, Literature.SummarizationFaithfulness, Literature.PrefixCaching
LIU, RULER = Literature.AnswerPositionAccuracy.entries, Literature.RulerEffectiveLength.entries

K = 10 ** 3
STEP_TURNS = 10  # hook step 2 pauses after this many turns (presentation choice)


def usd(q):
    return q.to("USD").magnitude


print("== Hook and Napkin Math 4.1 (repair request budget scenario)")
b = BUDGET.result
v("b_window", BUDGET.window, "{:,.0f}", "context window, tokens")
v("b_window_k", BUDGET.window / K, "{:.0f}", "context window, thousand tokens")
v("b_turns", BUDGET.turns, "{:.0f}", "step budget, turns")
v("b_reserve", BUDGET.output_reserve, "{:,.0f}", "output reserve, tokens")
v("b_reserve_k", BUDGET.output_reserve / K, "{:.0f}", "output reserve, thousand tokens")
v("b_system", BUDGET.system_tokens, "{:,.0f}", "system instructions, tokens")
v("b_tools", BUDGET.tool_tokens, "{:,.0f}", "tool declarations, tokens")
v("b_state", BUDGET.state_tokens, "{:,.0f}", "task state, tokens")
v("b_state_k", BUDGET.state_tokens / K, "{:.0f}", "task state, thousand tokens")
v("b_obs", BUDGET.observation_per_turn, "{:,.0f}", "tool result per turn, tokens")
v("b_digest", BUDGET.digest_tokens, "{:,.0f}", "digest per turn, tokens")
v("b_selected", BUDGET.selected_observation, "{:,.0f}", "selected observations per call, tokens")
v("b_limit", b.input_limit, "{:,.0f}", "input allowance, tokens")
v("b_limit_k", b.input_limit / K, "{:.0f}", "input allowance, thousand tokens")
v("b_root", BUDGET.system_tokens + BUDGET.tool_tokens, "{:,.0f}", "instructions plus tools, tokens")
v("b_root_k", (BUDGET.system_tokens + BUDGET.tool_tokens) / K, "{:.0f}", "the same, thousand tokens")
v("b_fixed", b.fixed_tokens, "{:,.0f}", "instructions, tools, and state, tokens")
v("b_fixed_k", b.fixed_tokens / K, "{:.0f}", "the same, thousand tokens")
v("b_obs_budget", b.observation_budget, "{:,.0f}", "observation budget, tokens")
v("b_turns_fit", b.turns_fit, "{:.0f}", "appended turns that fit")
v("b_overflow_turn", b.overflow_turn, "{:.0f}", "first turn that does not fit")
v("b_overflow_obs", b.overflow_observations, "{:,.0f}", "results the overflow turn needs, tokens")
v("b_naive_turns", BUDGET.window // BUDGET.observation_per_turn, "{:.0f}",
  "window / result size: the answer that forgets every reserve (vote choice D)")
v("b_max_obs", b.max_observation_per_turn, "{:,.0f}", "largest result that lasts every turn, tokens")
v("b_digest_turns", b.digest_turns, "{:.0f}", "appended turns that fit at the digest size")
v("b_step_turns", STEP_TURNS, "{:.0f}", "hook step 2: turns shown")
v("b_step_obs", STEP_TURNS * BUDGET.observation_per_turn, "{:,.0f}", "results after those turns, tokens")
v("b_append_bill", b.append_bill, "{:,.0f}", "appending, cumulative input, tokens")
v("b_select_bill", b.select_bill, "{:,.0f}", "selecting, cumulative input, tokens")
v("b_append_bill_m", b.append_bill / K ** 2, "{:.2f}", "appending, million tokens (figure)")
v("b_select_bill_m", b.select_bill / K ** 2, "{:.2f}", "selecting, million tokens (figure)")
v("b_ratio", b.bill_ratio, "{:.1f}", "append bill / select bill")
v("b_request", b.selected_request, "{:,.0f}", "selecting request, tokens")
v("b_final", b.final_append_request, "{:,.0f}", "appending, last request, tokens")
v("b_final_reserve", b.final_append_with_reserve, "{:,.0f}", "the same plus the output reserve")

print("== Recall (Lecture 2 resend scenario, Lecture 1 cost scenario)")
r30 = RESEND.result
v("rs_turns", RESEND.turns, "{:.0f}", "turns")
v("rs_first", RESEND.first_prompt, "{:,.0f}", "first prompt, tokens")
v("rs_delta", RESEND.appended, "{:,.0f}", "appended per turn, tokens")
v("rs_resent", r30.resent_tokens, "{:,.0f}", "resent input, tokens")
v("rs_fresh", r30.reused_prefix_tokens, "{:,.0f}", "fresh prefill with reuse, tokens")
v("rs_reduction", r30.reuse_reduction, "{:.1f}", "resent / fresh")
v("rec_attempt", COST.result.attempt_cost, "{:.2f}", "cost per attempt, $")
v("rec_acc", COST.acceptance, "{:.2f}", "acceptance probability")
v("rec_acc_pct", 100 * COST.acceptance, "{:.0f}", "acceptance probability, %")
v("rec_ceff", COST.result.cost_per_accepted, "{:.2f}", "cost per accepted task, $")

print("== Fiction.LiveBench sweep (Literature.LongContextComprehension)")
s = SWEEP.result
first, last = SWEEP.lengths[0], SWEEP.lengths[-1]
v("fl_models", s.scored_models, "{:.0f}", "models scored at every length")
v("fl_med_first", 100 * s.median[first], "{:.0f}", "median at the shortest version, %")
v("fl_med_8k", 100 * s.median["8k"], "{:.0f}", "median at 8K, %")
v("fl_med_last", 100 * s.median[last], "{:.0f}", "median at 60K, %")
v("fl_hold", s.held_count, "{:.0f}", "models at or above the bar at 60K")
v("fl_bar_pct", 100 * SWEEP.hold_bar, "{:.0f}", "the bar, %")
v("fl_hi", 100 * s.highest_last, "{:.0f}", "highest score at 60K, %")
v("fl_lo", 100 * s.lowest_last, "{:.0f}", "lowest score at 60K, %")

print("== RULER (Literature.RulerEffectiveLength, ContextUse)")
v("ruler_bar", USE.RulerEffectiveLengthThreshold, "{:.1f}", "effective-length threshold, %")
v("ruler_rows_words", len(RULER), "words", "rows plotted")
v("ruler_reached", sum(1 for e in RULER if e["effective_k"] is not None and e["effective_k"] >= e["claimed_k"]),
  "{:.0f}", "rows reaching their claimed length")
v("ruler_models", USE.RulerModelsEvaluated, "{:.0f}", "models the paper evaluated")

print("== Lost in the middle (Literature.AnswerPositionAccuracy, ContextUse)")
acc = {int(e["position"]): e["accuracy"] for e in LIU}
pos = sorted(acc)
worst = min(pos, key=lambda p: acc[p])
v("liu_docs", LIU[0]["documents"], "{:.0f}", "documents per question")
v("liu_first", acc[pos[0]], "{:.1f}", "answer first, %")
v("liu_last", acc[pos[-1]], "{:.1f}", "answer last, %")
v("liu_worst", acc[worst], "{:.1f}", "worst position, %")
v("liu_worst_pos", worst, "{:.0f}", "worst position")
v("liu_closed", USE.Gpt35TurboClosedBook, "{:.1f}", "closed book, %")

print("== Toy head (attention-dilution scenarios)")
h, x = HEAD.result, LOOK.result
v("dil_target", HEAD.target_logit, "{:g}", "target logit")
v("dil_background", HEAD.background_logit, "{:g}", "background logit")
v("dil_short", HEAD.short_keys, "{:,.0f}", "keys, short context")
v("dil_long", HEAD.long_keys, "{:,.0f}", "keys, long context")
v("dil_short_pct", 100 * h.short_weight, "{:.1f}", "target weight, short, %")
v("dil_long_pct", 100 * h.long_weight, "{:.1f}", "target weight, long, %")
v("dil_unrelated", LOOK.unrelated_keys, "{:,.0f}", "unrelated keys")
v("dil_look", LOOK.lookalike_keys, "{:,.0f}", "look-alike keys (stale excerpt tokens)")
v("dil_look_logit", LOOK.lookalike_logit, "{:g}", "look-alike logit")
v("dil_gap", LOOK.lookalike_logit - LOOK.background_logit, "{:g}", "look-alike minus background logit")
v("dil_without_pct", 100 * x.weight_without_lookalikes, "{:.1f}", "without the excerpt, %")
v("dil_with_pct", 100 * x.weight_with_lookalikes, "{:.1f}", "with the excerpt, %")
v("dil_equiv", x.lookalike_equivalent_keys, "{:,.0f}", "unrelated keys equal to the excerpt")
v("dil_half_pct", 100 * x.weight_half_unrelated, "{:.1f}", "half the unrelated keys dropped, %")

print("== Napkin Math 4.2 (prefix layout bill scenarios)")
L, L4 = LAYOUT.result, LAYOUT4.result
v("lay_root", LAYOUT.root_tokens, "{:,.0f}", "root, tokens")
v("lay_root_k", LAYOUT.root_tokens / K, "{:.0f}", "root, thousand tokens")
v("lay_trunk", LAYOUT.trunk_tokens, "{:,.0f}", "trunk, tokens")
v("lay_trunk_k", LAYOUT.trunk_tokens / K, "{:.0f}", "trunk, thousand tokens")
v("lay_leaf", LAYOUT.leaf_tokens, "{:,.0f}", "leaf, tokens")
v("lay_leaf_k", LAYOUT.leaf_tokens / K, "{:.0f}", "leaf, thousand tokens")
v("lay_calls", LAYOUT.turns, "{:.0f}", "calls")
v("lay_rewrites", LAYOUT.trunk_rewrites, "{:.0f}", "trunk rewrites")
v("lay_rewrites_words", LAYOUT.trunk_rewrites, "words", "trunk rewrites, in words")
v("lay_price", LAYOUT.input_price.to("USD / Mtoken").magnitude, "{:.0f}", "input price, $ per million tokens")
v("lay_cached_pct", 100 * LAYOUT.cached_fraction, "{:.0f}", "cached reads billed at, %")
v("lay_request", L.request_tokens, "{:,.0f}", "request, tokens")
v("lay_total", L.total_input, "{:,.0f}", "trajectory input, tokens")
v("lay_rewrite", L.rewrite_tokens, "{:,.0f}", "a trunk rewrite prefills, tokens")
v("lay_warm", L.warm_calls, "{:.0f}", "leaf-only calls")
v("lay_uncached", L.uncached_tokens, "{:,.0f}", "stable layout, uncached tokens")
v("lay_cached", L.cached_tokens, "{:,.0f}", "stable layout, cached tokens")
v("lay_hit_pct", 100 * L.hit_rate, "{:.1f}", "cached share, %")
v("lay_unstable", usd(L.unstable_cost), "{:.2f}", "volatile-first bill, $")
v("lay_stable", usd(L.stable_cost), "{:.2f}", "stable bill, $")
v("lay_ratio", L.cost_ratio, "{:.1f}", "bill ratio")
v("lay_rewrite_calls", L.rewrite_in_leaf_calls, "{:.0f}", "a trunk rewrite in leaf-only calls")
v("lay4_rewrites", LAYOUT4.trunk_rewrites, "{:.0f}", "Checkpoint 4.1: rewrites on every fourth call")
v("lay4_uncached", L4.uncached_tokens, "{:,.0f}", "Checkpoint 4.1: uncached tokens")
v("lay4_stable", usd(L4.stable_cost), "{:.2f}", "Checkpoint 4.1: stable bill, $")

print("== SGLang deployment (Literature.PrefixCaching)")
v("sg_llava", CACHE.SglangLlavaNext34BHitRate, "{:.1f}", "LLaVA-Next-34B hit rate, %")
v("sg_vicuna", CACHE.SglangVicuna33BHitRate, "{:.1f}", "Vicuna-33B hit rate, %")
v("sg_ttft", CACHE.SglangVicuna33BFirstTokenSpeedup, "{:.1f}", "Vicuna-33B first-token latency reduction, x")

print("== Tokenizer run (artifacts/tokenize_output.txt)")
runs = re.findall(r"shared prefix (\d+) of (\d+)", (HERE / "artifacts" / "tokenize_output.txt").read_text())
v("tok_len", int(runs[0][1]), "{:.0f}", "request tokens")
v("tok_shared_first", int(runs[0][0]), "{:.0f}", "shared prefix, id first")
v("tok_shared_last", int(runs[1][0]), "{:.0f}", "shared prefix, id last")

print("== Summaries (Literature.SummarizationFaithfulness)")
systems = (SUMM.XsumPtGenHallucinated, SUMM.XsumTConvS2SHallucinated, SUMM.XsumTranS2SHallucinated,
           SUMM.XsumBertS2SHallucinated)
v("mz_n", SUMM.XsumSummariesPerSystem, "{:.0f}", "summaries per system")
v("mz_systems_words", len(systems), "words", "model summarizers")
v("mz_lo", min(systems), "{:.1f}", "lowest model share, %")
v("mz_hi", max(systems), "{:.1f}", "highest model share, %")
v("mz_gold", SUMM.XsumGoldHallucinated, "{:.1f}", "reference summaries, %")

print("== Napkin Math 4.3 (compaction watermark scenario)")
w = MARKS.result
v("cw_target", MARKS.target, "{:,.0f}", "target, tokens")
v("cw_target_k", MARKS.target / K, "{:.0f}", "target, thousand tokens")
v("cw_trigger", MARKS.trigger, "{:,.0f}", "trigger, tokens")
v("cw_trigger_k", MARKS.trigger / K, "{:.0f}", "trigger, thousand tokens")
v("cw_gap", MARKS.trigger - MARKS.target, "{:,.0f}", "watermark gap, tokens")
v("cw_cycle", w.cycle_turns, "{:.0f}", "turns between passes")
v("cw_passes", w.passes, "{:.0f}", "passes in the step budget")
v("cw_passes_words", w.passes, "words", "the same, in words")
v("cw_trigger_len", w.trigger_length, "{:,.0f}", "request when a pass fires, tokens")
v("cw_overshoot", w.overshoot, "{:,.0f}", "overshoot, tokens")
v("cw_with_reserve", w.length_with_reserve, "{:,.0f}", "the same plus the reserve, tokens")
v("cw_alt_target", MARKS.alternate_target, "{:,.0f}", "narrow target, tokens")
v("cw_alt_target_k", MARKS.alternate_target / K, "{:.0f}", "narrow target, thousand tokens")
v("cw_alt_gap", MARKS.trigger - MARKS.alternate_target, "{:,.0f}", "narrow gap, tokens")
v("cw_alt_cycle", w.alternate_cycle_turns, "{:.0f}", "narrow: turns between passes")
v("cw_alt_passes", w.alternate_passes, "{:.0f}", "narrow: passes")
v("cw_freq", w.frequency_ratio, "{:.0f}", "narrow / wide pass frequency")

print("== Equation 6 and Exercise 6 (retention and protected-field check scenarios)")
rt, c = RETAIN.result, CHECK.result
v("ret_loss_pct", 100 * rt.step_error, "{:.0f}", "loss per round, %")
v("ret_loss", rt.step_error, "{:.2f}", "loss per round")
v("ret_short", RETAIN.short_horizon, "{:.0f}", "short run, rounds")
v("ret_long", RETAIN.long_horizon, "{:.0f}", "long run, rounds")
v("ret_short_pct", 100 * rt.success[RETAIN.short_horizon], "{:.1f}", "survives the short run, %")
v("ret_long_pct", 100 * rt.success[RETAIN.long_horizon], "{:.1f}", "survives the long run, %")
v("ret_below", rt.first_horizon_below_target, "{:.0f}", "first round below even odds")
v("ret_floor_pct", 100 * RETAIN.target_success, "{:.0f}", "even odds, %")
v("pf_constraints", CHECK.constraints, "{:.0f}", "constraints")
v("pf_constraints_words", CHECK.constraints, "words", "constraints, in words")
v("pf_loss_pct", 100 * CHECK.loss_per_round, "{:.0f}", "loss per constraint per round, %")
v("pf_rounds", CHECK.rounds, "{:.0f}", "rounds")
v("pf_pacc", CHECK.acceptance_if_intact, "{:.1f}", "acceptance when all survive")
v("pf_attempt", usd(CHECK.attempt_cost), "{:.0f}", "attempt cost, $")
v("pf_detect_pct", 100 * CHECK.detection, "{:.0f}", "losses the check catches, %")
v("pf_check_cost", usd(CHECK.check_cost_per_round), "{:.2f}", "check cost per round, $")
v("pf_s_no_pct", 100 * c.survival_unchecked, "{:.1f}", "all survive, no check, %")
v("pf_s_no", c.survival_unchecked, "{:.3f}", "the same, fraction")
v("pf_p_no", c.acceptance_unchecked, "{:.3f}", "p_acc, no check")
v("pf_c_no", c.cost_per_accepted_unchecked, "{:.2f}", "C_eff, no check, $")
v("pf_undetected_pct", 100 * c.undetected_loss, "{:.1f}", "loss the check misses, % per round")
v("pf_s_chk_pct", 100 * c.survival_checked, "{:.1f}", "all survive, with check, %")
v("pf_p_chk", c.acceptance_checked, "{:.3f}", "p_acc, with check")
v("pf_attempt_chk", c.attempt_cost_checked, "{:.2f}", "attempt with checks, $")
v("pf_c_chk", c.cost_per_accepted_checked, "{:.2f}", "C_eff, with check, $")

print("== Napkin Math 4.4 (invalidation scenario) and Shi et al. 2023 (ContextUse)")
iv = INVAL.result
v("iv_stale", INVAL.superseded_tokens, "{:,.0f}", "superseded excerpt, tokens")
v("iv_marker", INVAL.marker_tokens, "{:,.0f}", "marker, tokens")
v("iv_notice", INVAL.notice_tokens, "{:,.0f}", "notice, tokens")
v("iv_prefix", INVAL.prefix_tokens, "{:,.0f}", "reused root, tokens")
v("iv_replaced_suffix", iv.replaced_suffix, "{:,.0f}", "replace: prefilled again, tokens")
v("iv_appended_suffix", iv.appended_suffix, "{:,.0f}", "append: prefilled, tokens")
v("iv_ratio", iv.suffix_ratio, "{:.0f}", "replace / append")
base = USE.CodeDavinci002CotBase
noisy = USE.CodeDavinci002CotIrrelevantSentence
told = USE.CodeDavinci002CotIrrelevantSentenceIgnoreInstruction
v("shi_base", base, "{:.1f}", "base problems, %")
v("shi_noisy", noisy, "{:.1f}", "one irrelevant sentence, %")
v("shi_told", told, "{:.1f}", "plus an instruction to ignore it, %")
v("shi_drop", base - noisy, "{:.1f}", "points lost")
v("shi_recovered", told - noisy, "{:.1f}", "points recovered")

if __name__ == "__main__":
    write(__file__)
