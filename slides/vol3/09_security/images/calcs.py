#!/usr/bin/env python3
"""Every number on the Volume III Lecture 9 slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations, and the chapter reads the same
entries: Napkin Math 9.1 and the Lighthouse 9.1 table are
Scenarios.InjectionExposureByReadVolume, equation 5's turn form is
Scenarios.RepairTrajectoryTurnExposure, Napkin Math 9.2 is
Scenarios.FixtureReleasePolicyComparison (attempts priced at the introduction's
Scenarios.RepairAttemptCostLevers), the zero-success bound is
Scenarios.CleanAttackSuite50 and CleanAttackSuite100, the prefix-cache timing
is Scenarios.PrefixCacheProbe2K and InvariantPrefixRebuildDgxH100, and the
fifty-repository example is Scenarios.DelegatedRepositoryGrant. Cited results
are Literature entries: AgentDojo's published results table
(vol3/09_security/data/agentdojo_results.csv) and the prompt-cache audit of Gu et al.,
ICML 2025 (Literature.PromptCacheAudit). This file only selects values,
converts units, and formats them; it types no numbers. Each value is recorded
with v() under the key the deck prints as \\val{key}; running this file writes
images/numbers.tex. make_figs.py imports it so figures and slides agree.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402

Literature, Scenarios = mlsysim.Literature, mlsysim.Scenarios

EXPOSURE = Scenarios.InjectionExposureByReadVolume
TURNS = Scenarios.RepairTrajectoryTurnExposure
GRANT = Scenarios.DelegatedRepositoryGrant
PROBE = Scenarios.PrefixCacheProbe2K
REBUILD = Scenarios.InvariantPrefixRebuildDgxH100
AUDIT = Literature.PromptCacheAudit
POLICY = Scenarios.FixtureReleasePolicyComparison
CLEAN50 = Scenarios.CleanAttackSuite50
CLEAN100 = Scenarios.CleanAttackSuite100
DOJO = load_series("vol3/09_security/data/agentdojo_results.csv").entries

# AgentDojo rows the deck names (the published table's model and defense ids)
ATTACK = "important_instructions"
ABLATED = "gpt-4o-2024-05-13"
EARLY, LATE = "claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20241022"
DEFENSES = ["spotlighting_with_delimiting", "transformers_pi_detector",
            "tool_filter", "repeat_user_prompt"]


def dojo_row(model, defense="none"):
    return next(r for r in DOJO if r["model"] == model and r["defense"] == defense
                and r["attack"] == ATTACK)


DOJO_PLAIN = [r for r in DOJO if r["defense"] == "none" and r["attack"] == ATTACK]

print("== Napkin Math 9.1 and Lighthouse 9.1 (Scenarios.InjectionExposureByReadVolume)")
e = EXPOSURE.result
v("exp_q", EXPOSURE.payload_fraction * 100, "{:.0f}", "payload fraction, %")
v("exp_miss", EXPOSURE.miss_rate * 100, "{:.0f}", "screen miss rate, %")
v("exp_miss_frac", EXPOSURE.miss_rate, "{:.2f}", "screen miss rate, fraction")
for k in ("repair", "research", "support", "remediation"):
    v(f"exp_n_{k}", EXPOSURE.reads[k], "{:.0f}", f"{k} reads per task")
    v(f"exp_{k}", e.exposure[k] * 100, "{:.1f}", f"{k} exposure, %")
v("exp_repair_lin", e.linear["repair"] * 100, "{:.0f}", "repair linear estimate q n, %")
v("exp_repair_scr", e.screened["repair"] * 100, "{:.1f}", "repair behind the screen, %")
v("exp_research_scr", e.screened["research"] * 100, "{:.1f}", "research behind the screen, %")
v("exp_cohort", EXPOSURE.cohort, "{:,.0f}", "research tasks in the cohort")
v("exp_research_cohort", e.screened_per_cohort["research"], "{:.0f}", "screened research tasks exposed per cohort")

print("== Equation 5 in turns (Scenarios.RepairTrajectoryTurnExposure)")
t = TURNS.result
v("tn_reads", TURNS.reads_per_turn, "{:.0f}", "untrusted reads per turn")
v("tn_reads_words", TURNS.reads_per_turn, "words", "the same, in words")
v("tn_turns", TURNS.turns, "{:.0f}", "turns (the repair harness's step budget)")
v("tn_per_turn", t.per_turn * 100, "{:.1f}", "per-turn exposure, %")
v("tn_traj", t.trajectory * 100, "{:.1f}", "trajectory exposure, %")
v("tn_docs", t.reads, "{:.0f}", "the same reads counted as documents")
v("tn_rest", TURNS.turns - 1, "{:.0f}", "turns after the first")

print("== Effective permission (Scenarios.DelegatedRepositoryGrant)")
v("gr_held", GRANT.held, "{:.0f}", "repositories the requester can read")
v("gr_needed", GRANT.needed, "{:.0f}", "repositories the fix needs")
v("gr_refused", GRANT.refused, "{:.0f}", "repositories the broker refuses")

print("== Prefix-cache timing (Scenarios.PrefixCacheProbe2K, InvariantPrefixRebuildDgxH100)")
v("pp_short", PROBE.prefix_tokens, "{:,.0f}", "probed prefix, tokens")
v("pp_miss_short", PROBE.result.rebuild_time.to("ms").magnitude, "{:.0f}", "miss at the probed prefix, ms")
v("pp_hit", PROBE.result.weight_read_time.to("ms").magnitude, "{:.0f}", "hit (weight-read pass), ms")
v("pp_long", REBUILD.prefix_tokens, "{:,.0f}", "longer prefix, tokens")
v("pp_miss_long", REBUILD.result.rebuild_time.to("ms").magnitude, "{:.0f}", "miss at the longer prefix, ms")

print("== Prompt-cache audit (Literature.PromptCacheAudit, Gu et al. 2025)")
v("gu_audited", AUDIT.ProvidersAudited, "{:.0f}", "providers audited")
v("gu_caching", AUDIT.ProvidersCaching, "{:.0f}", "providers caching")
v("gu_global", AUDIT.ProvidersGlobalSharing, "{:.0f}", "providers sharing globally")
v("gu_global_words", AUDIT.ProvidersGlobalSharing, "words", "the same, in words")

print("== AgentDojo, important_instructions attack (vol3/09_security/data/agentdojo_results.csv)")
early, late, base = dojo_row(EARLY), dojo_row(LATE), dojo_row(ABLATED)
v("dj_models", len(DOJO_PLAIN), "{:.0f}", "undefended models")
v("dj_asr_lo", min(r["targeted_asr_pct"] for r in DOJO_PLAIN), "{:.0f}", "lowest attack success, %")
v("dj_asr_hi", max(r["targeted_asr_pct"] for r in DOJO_PLAIN), "{:.0f}", "highest attack success, %")
assert early["utility_pct"] == late["utility_pct"], "the two versions share one utility"
v("dj_same_util", early["utility_pct"], "{:.0f}", "Claude 3.5 Sonnet utility, both versions, %")
v("dj_early", early["targeted_asr_pct"], "{:.0f}", "June 2024 version attack success, %")
v("dj_late", late["targeted_asr_pct"], "{:.0f}", "October 2024 version attack success, %")
v("dj_ratio_floor", math.floor(early["targeted_asr_pct"] / late["targeted_asr_pct"] / 10) * 10, "{:.0f}",
  "early / late attack success, floored to tens (\"more than\")")
v("dj_base_util", base["utility_pct"], "{:.0f}", "GPT-4o undefended utility, %")
v("dj_base_asr", base["targeted_asr_pct"], "{:.0f}", "GPT-4o undefended attack success, %")
for key, d in (("spot", "spotlighting_with_delimiting"), ("det", "transformers_pi_detector"),
               ("filt", "tool_filter"), ("rep", "repeat_user_prompt")):
    r = dojo_row(ABLATED, d)
    v(f"dj_{key}_util", r["utility_pct"], "{:.0f}", f"GPT-4o + {d} utility, %")
    v(f"dj_{key}_asr", r["targeted_asr_pct"], "{:.0f}", f"GPT-4o + {d} attack success, %")

print("== Napkin Math 9.2 (Scenarios.FixtureReleasePolicyComparison)")
p = POLICY.result
v("pc_attacked", POLICY.attacked, "{:.0f}", "attacked episodes per policy")
v("pc_benign", POLICY.benign, "{:.0f}", "benign episodes per policy")
v("pc_a_lo", p.baseline_lower * 100, "{:.0f}", "A confirmed, %")
v("pc_b_lo", p.candidate_lower * 100, "{:.0f}", "B confirmed, %")
v("pc_red", p.reduction_observed * 100, "{:.0f}", "relative reduction, observed, %")
v("pc_a_hi", p.baseline_upper * 100, "{:.0f}", "A with unresolved as disclosure, %")
v("pc_b_hi", p.candidate_upper * 100, "{:.0f}", "B with unresolved as disclosure, %")
v("pc_red_hi", p.reduction_upper * 100, "{:.0f}", "relative reduction, both upper, %")
v("pc_red_lf", p.reduction_least_favorable * 100, "{:.0f}", "least favorable reduction, %")
v("pc_acc_a", p.baseline_acceptance, "{:.2f}", "benign acceptance A")
v("pc_acc_b", p.candidate_acceptance, "{:.2f}", "benign acceptance B")
v("pc_acc_a_n", POLICY.baseline["accepted_benign"], "{:.0f}", "A accepted benign repairs")
v("pc_acc_b_n", POLICY.candidate["accepted_benign"], "{:.0f}", "B accepted benign repairs")
v("pc_extra_fb", p.extra_false_blocks, "{:.0f}", "B's extra false blocks")
v("pc_rise", p.cost_rise * 100, "{:.1f}", "rise in cost per accepted benign repair, %")
v("pc_attempt", POLICY.attempt_cost, "{:.2f}", "attempt cost, $")
v("pc_ceff_a", p.baseline_cost_per_accepted, "{:.2f}", "C_eff A, $")
v("pc_ceff_b", p.candidate_cost_per_accepted, "{:.2f}", "C_eff B, $")

print("== Zero-success bound (Scenarios.CleanAttackSuite50, CleanAttackSuite100)")
z50, z100 = CLEAN50.result, CLEAN100.result
v("zb_conf", CLEAN50.confidence * 100, "{:.0f}", "confidence, %")
v("zb_alpha", 1 - CLEAN50.confidence, "{:.2f}", "1 - confidence")
v("zb_n50", CLEAN50.trials, "{:.0f}", "clean attacked episodes (checkpoint)")
v("zb_ub50", z50.rate_upper_exact * 100, "{:.1f}", "95% upper bound at 50, %")
v("zb_n100", CLEAN100.trials, "{:.0f}", "clean attacked episodes (chapter prose)")
v("zb_ub100", z100.rate_upper_exact * 100, "{:.1f}", "95% upper bound at 100, %")
v("zb_target", CLEAN50.target_rate * 100, "{:.0f}", "tolerable attack-success rate, %")
v("zb_need", z50.trials_for_target, "{:.0f}", "clean episodes the target needs")
v("zb_rule", z50.trials_for_target_approx, "{:.0f}", "rule-of-three estimate")

if __name__ == "__main__":
    write(__file__)
