#!/usr/bin/env python3
"""Every number on the Volume III Lecture 17 slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations, and the chapter reads the same
entries: the verifier's false accepts (VerifierFalseAcceptsAtTenPercentPass and
its 5 and 40 percent variants), fixed baselines (BinaryRewardFixedBaseline,
BinaryRewardAveragePassBaseline), shaped rewards (ShapedRewardOverlappingWeights,
ShapedRewardOrderedWeights), forked continuations (DebuggerTurnForkValue,
ThirtyTurnForksAtEveryBoundary and its two-fork variant), group-relative
coefficients (RepairGroupOnePassOfThree and its TRL-convention variant,
LonePassInGroupOfSixteen), length normalization (ShortAndLongFailureLengths),
reward contrast per update (HardTaskRolloutStep and its variants), the all-pass
rule (FlakyRaceTestAllPassRule), the ratio clip (DominantTokenRatioClip,
RareTokenRatioClip), the slot cycle (RepairRolloutSlotResets), the group
barrier (RolloutGroupOfFourBarrier), and the verifier pool (RepairVerifierPool),
backed by engine/solvers/reinforcement.py. Lecture 16's target mask is
RecoveryRecordTargetMask. Published data are Literature entries: METR's
reward-hacking counts (Literature.AgentEvaluation), Yue et al.'s pass@k series
(slides/vol3/17_rlvr/images/data/yue_grpo_pass_at_k.csv), and the DeepSeek-R1-Zero and DAPO curve readings
(Literature.RlTrainingDynamics). The one other source is the deck's own data
(data/): a local model's recorded log probabilities. This file only selects
values, converts units, and formats them; it types no numbers. Each value is
recorded with v() under the key the deck prints as \\val{key}; running this
file writes images/numbers.tex. make_figs.py imports the curve functions below.

Run from this folder: python3 calcs.py
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402

Scenarios = mlsysim.Scenarios
Literature = mlsysim.Literature
solvers = mlsysim.solvers
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

METR = Literature.AgentEvaluation
RLD = Literature.RlTrainingDynamics
YUE = sorted(load_series("slides/vol3/17_rlvr/images/data/yue_grpo_pass_at_k.csv").entries, key=lambda r: r["step"])

FA_S = Scenarios.VerifierFalseAcceptsAtTenPercentPass
FA5_S = Scenarios.VerifierFalseAcceptsAtFivePercentPass
FA40_S = Scenarios.VerifierFalseAcceptsAtFortyPercentPass
BASE_S = Scenarios.BinaryRewardAveragePassBaseline
SHAPE_S = Scenarios.ShapedRewardOverlappingWeights
SHAPE_OK_S = Scenarios.ShapedRewardOrderedWeights
FORK_S = Scenarios.DebuggerTurnForkValue
EVERY_S = Scenarios.ThirtyTurnForksAtEveryBoundary
TWO_S = Scenarios.ThirtyTurnForksAfterTurnsTenAndTwenty
GRPO_S = Scenarios.RepairGroupOnePassOfThree
TRL_S = Scenarios.RepairGroupOnePassOfThreeTrlConvention
LONE_S = Scenarios.LonePassInGroupOfSixteen
LEN_S = Scenarios.ShortAndLongFailureLengths
HARD_S = Scenarios.HardTaskRolloutStep
BIG_S = Scenarios.HardTaskRolloutStepDoubledGroup
TGT_S = Scenarios.TargetedTaskRolloutStep
PAIR_S = Scenarios.PairGroupsAtTenPercentPass
FLAKY_S = Scenarios.FlakyRaceTestAllPassRule
CLIP_HI_S = Scenarios.DominantTokenRatioClip
CLIP_LO_S = Scenarios.RareTokenRatioClip
SLOT_S = Scenarios.RepairRolloutSlotResets
BAR_S = Scenarios.RolloutGroupOfFourBarrier
POOL_S = Scenarios.RepairVerifierPool
MASK = Scenarios.RecoveryRecordTargetMask.result


def pct(x):
    return 100 * x


def sec(q):
    return q.to("s").magnitude


def ms(q):
    return q.to("ms").magnitude


# ---------------------------------------------------------------- curves for make_figs.py
def false_share(p):
    """Share of accepted rollouts that failed, at pass rate p, under the deck's verifier."""
    return FA_S.variant(pass_rate=p).result.false_share


def contrast(p, group):
    """P(a group of `group` rollouts at pass rate p holds a pass and a failure)."""
    return HARD_S.variant(group=group, pass_rate=p).result.contrast


def coefficients(rewards):
    """GRPO coefficients of one group under the worked example's convention."""
    return solvers.GroupCoefficientsModel().solve(rewards, GRPO_S.stabilizer).coefficients


def lone_pass(group):
    """The lone pass's coefficient in a group of `group` binary rewards, stabilizer omitted."""
    rewards = [1.0] + [0.0] * (group - 1)
    return solvers.GroupCoefficientsModel().solve(rewards, LONE_S.stabilizer).coefficients[0]


def all_pass(trials):
    """(correct accepted, incorrect accepted) when all `trials` must pass."""
    r = FLAKY_S.variant(trials=trials).result
    return 1 - r.correct_rejected, r.incorrect_accepted


print("== Hook and prompt table (Literature.AgentEvaluation, METR 2025)")
METR_TASKS = [  # (label, hacks, runs, kind) in the task table's order
    ("LLM Foundry", METR.MetrO3FoundryHackRuns, METR.MetrO3FoundryRuns, "task"),
    ("Rust scaffold", METR.MetrO3RustScaffoldHackRuns, METR.MetrO3RustScaffoldRuns, "task"),
    ("Kernel", METR.MetrO3KernelHackRuns, METR.MetrO3KernelRuns, "task"),
    ("RE-Bench, all", METR.MetrO3ReBenchHackRuns, METR.MetrO3ReBenchRuns, "rebench"),
    ("HCAST, all", METR.MetrO3HcastHackRuns, METR.MetrO3HcastRuns, "hcast"),
]
METR_TASK_ROWS = [(label, h, n, kind, pct(h / n)) for label, h, n, kind in METR_TASKS]
METR_PROMPTS = [  # (label, percent planning a hack)
    ("Original prompt", METR.MetrO3PromptOriginalHackPct),
    ("Designer's intent", METR.MetrO3PromptDesignerIntentHackPct),
    ("Do not cheat", METR.MetrO3PromptDoNotCheatHackPct),
    ("Do not reward hack", METR.MetrO3PromptDoNotRewardHackPct),
    ("Alzheimer's stakes", METR.MetrO3PromptResearchStakesHackPct),
]
v("metr_rebench_hacks", METR.MetrO3ReBenchHackRuns, "{:.0f}", "RE-Bench runs that hacked")
v("metr_rebench_runs", METR.MetrO3ReBenchRuns, "{:.0f}", "RE-Bench runs")
v("metr_rebench_pct", pct(METR.MetrO3ReBenchHackRuns / METR.MetrO3ReBenchRuns), "{:.0f}", "RE-Bench share, %")
v("metr_hcast_hacks", METR.MetrO3HcastHackRuns, "{:.0f}", "HCAST runs that hacked")
v("metr_hcast_runs", METR.MetrO3HcastRuns, "{:,.0f}", "HCAST runs")
v("metr_hcast_pct", pct(METR.MetrO3HcastHackRuns / METR.MetrO3HcastRuns), "{:.1f}", "HCAST share, %")
v("metr_foundry_runs", METR.MetrO3FoundryRuns, "{:.0f}", "Foundry runs, all hacked")
v("metr_intent_no", METR.MetrO3IntentNoAnswers, "{:.0f}", "answered no")
v("metr_intent_asked", METR.MetrO3IntentQuestions, "{:.0f}", "times asked")
v("metr_prompt_runs", METR.MetrO3PromptVariantRuns, "{:.0f}", "runs per prompt")
v("metr_base_pct", METR.MetrO3PromptOriginalHackPct, "{:.0f}", "original prompt, %")
v("metr_intent_pct", METR.MetrO3PromptDesignerIntentHackPct, "{:.0f}", "designer's intent, %")
v("metr_cheat_pct", METR.MetrO3PromptDoNotCheatHackPct, "{:.0f}", "do not cheat, %")
v("metr_nohack_pct", METR.MetrO3PromptDoNotRewardHackPct, "{:.0f}", "do not reward hack, %")
v("metr_stakes_pct", METR.MetrO3PromptResearchStakesHackPct, "{:.0f}", "research stakes, %")

print("== Recall: Lecture 16's target mask (Scenarios.RecoveryRecordTargetMask)")
v("mask_obs_pct", pct(MASK.observation_share), "{:.1f}", "observation share, %")
v("mask_tgt_pct", pct(MASK.share_without_rationale), "{:.1f}", "selected targets, %")

print("== False accepts (Scenarios.VerifierFalseAcceptsAtTenPercentPass and variants)")
FA = FA_S.result
v("fa_tasks", FA_S.tasks, "{:,.0f}", "tasks")
v("fa_group", FA_S.group, "{:.0f}", "rollouts per task")
v("fa_rollouts", FA.rollouts, "{:,.0f}", "rollouts")
v("fa_rate", pct(FA_S.false_accept_rate), "{:.1f}", "false-acceptance rate, %")
v("fa_pass", pct(FA_S.pass_rate), "{:.0f}", "pass rate, %")
v("fa_true", FA.true_accepts, "{:.1f}", "true accepts")
v("fa_false", FA.false_accepts, "{:.2f}", "false accepts")
v("fa_share", pct(FA.false_share), "{:.1f}", "false share at 10%, %")
v("fa5_pass", pct(FA5_S.pass_rate), "{:.0f}", "lower pass rate, %")
v("fa5_share", pct(FA5_S.result.false_share), "{:.1f}", "false share at 5%, %")
v("fa40_pass", pct(FA40_S.pass_rate), "{:.0f}", "higher pass rate, %")
v("fa40_share", pct(FA40_S.result.false_share), "{:.1f}", "false share at 40%, %")

print("== Shaping (Scenarios.ShapedRewardOrderedWeights and ShapedRewardOverlappingWeights)")
SH = SHAPE_S.result
v("shape_ok_alpha", SHAPE_OK_S.format_weight, "{:.1f}", "ordered alpha")
v("shape_ok_lambda", SHAPE_OK_S.length_weight, "{:.1f}", "ordered lambda")
v("shape_alpha", SHAPE_S.format_weight, "{:.1f}", "Exercise 2 alpha")
v("shape_lambda", SHAPE_S.length_weight, "{:.1f}", "Exercise 2 lambda")
v("shape_sum", SH.weight_sum, "{:.1f}", "alpha + lambda")
v("shape_low_pass", SH.lowest_pass, "{:.2f}", "lowest pass, 1 - lambda")
v("shape_high_fail", SH.highest_failure, "{:.2f}", "highest failure, alpha")
v("shape_pass_tokens", SHAPE_S.pass_tokens, "{:,.0f}", "example pass, tokens")
v("shape_pass_format", SHAPE_S.pass_format, "{:.1f}", "example pass, format score")
v("shape_pass_score", SH.example_pass, "{:.3f}", "example pass, score")
v("shape_fail_tokens", SHAPE_S.failure_tokens, "{:.0f}", "example failure, tokens")
v("shape_fail_score", SH.example_failure, "{:.3f}", "example failure, score")

print("== Credit (Scenarios.BinaryRewardAveragePassBaseline, DebuggerTurnForkValue)")
FK = FORK_S.result
v("credit_base", BASE_S.baseline, "{:.1f}", "baseline")
v("credit_fail", BASE_S.result.failing_score, "{:.1f}", "failed rollout coefficient")
v("fork_k", FORK_S.continuations, "{:.0f}", "continuations per fork")
v("fork_before", FORK_S.passes_before, "{:.0f}", "passes before the debugger turn")
v("fork_after", FORK_S.passes_after, "{:.0f}", "passes after")
v("fork_v0", FK.value_before, "{:.2f}", "value before")
v("fork_v1", FK.value_after, "{:.2f}", "value after")
v("fork_diff", FK.difference, "{:.2f}", "difference")
v("fork_bound", FK.resolution_threshold, "{:.2f}", "noise bound (two difference SEs)")

print("== Fork cost (Scenarios.ThirtyTurnForksAtEveryBoundary and its two-fork variant)")
EV, TW = EVERY_S.result, TWO_S.result
v("forks_turns", EVERY_S.turns, "{:.0f}", "turns")
v("forks_tokens", EVERY_S.tokens_per_turn, "{:.0f}", "tokens per turn")
v("forks_branches", EVERY_S.branches, "{:.0f}", "continuations per boundary")
v("forks_base", EV.base_tokens, "{:,.0f}", "base tokens")
v("forks_all_branches", EV.branches, "{:.0f}", "branches, every boundary")
v("forks_all_tokens", EV.total_tokens, "{:,.0f}", "tokens, every boundary")
v("forks_all_mult", EV.multiple, "{:.0f}", "multiple, every boundary")
v("forks_two_first", TWO_S.boundaries[0], "{:.0f}", "first chosen boundary")
v("forks_two_second", TWO_S.boundaries[1], "{:.0f}", "second chosen boundary")
v("forks_two_branches", TW.branches, "{:.0f}", "branches, two forks")
v("forks_two_mult", TW.multiple, "{:.0f}", "multiple, two forks")

print("== GRPO coefficients (Scenarios.RepairGroupOnePassOfThree, TRL variant, lone pass)")
G3, TRL, LONE = GRPO_S.result, TRL_S.result, LONE_S.result
v("grpo_eps_exp", math.log10(GRPO_S.stabilizer), "{:.0f}", "stabilizer, power of ten")
v("grpo_mean", G3.mean, "{:.6f}", "mean")
v("grpo_std", G3.std, "{:.6f}", "std, divisor G")
v("grpo_std3", G3.std, "{:.3f}", "the same, three decimals")
v("grpo_fail", G3.coefficients[0], "{:.6f}", "failure coefficient")
v("grpo_fail3", G3.coefficients[0], "{:.3f}", "the same, three decimals")
v("grpo_pass", G3.coefficients[1], "{:.6f}", "pass coefficient")
v("grpo_pass3", G3.coefficients[1], "{:.3f}", "the same, three decimals")
v("lone_g", LONE.group, "{:.0f}", "group size")
v("lone_fails", LONE.group - 1, "{:.0f}", "failures in the group")
v("lone_fails_words", LONE.group - 1, "words", "the same, in words")
v("lone_pass", LONE.coefficients[0], "{:.2f}", "lone-pass coefficient")
v("lone_fail", LONE.coefficients[1], "{:.2f}", "each failure's coefficient")
v("trl_eps_exp", math.log10(TRL_S.stabilizer), "{:.0f}", "TRL stabilizer, power of ten")
v("trl_pass", TRL.coefficients[1], "{:.3f}", "TRL pass coefficient")
v("trl_fail", TRL.coefficients[0], "{:.3f}", "TRL failure coefficient")

print("== Length (Scenarios.ShortAndLongFailureLengths; Literature.RlTrainingDynamics)")
v("len_short", LEN_S.short_tokens, "{:.0f}", "short failure, tokens")
v("len_long", LEN_S.long_tokens, "{:,.0f}", "long failure, tokens")
v("len_ratio", LEN_S.result.per_token_ratio, "{:.0f}", "per-token push ratio")
v("r1_len_start", RLD.R1ZeroResponseLengthStartBelowTokens, "{:,.0f}", "R1-Zero start, under, tokens")
v("r1_len_end", RLD.R1ZeroResponseLengthEndAboutTokens, "{:,.0f}", "R1-Zero end, about, tokens")
v("r1_steps", RLD.R1ZeroPlottedStepsAbout, "{:,.0f}", "R1-Zero steps plotted, about")

print("== Contrast and rollout budget (Scenarios.HardTaskRolloutStep and variants)")
HD, BG, TG, PR = HARD_S.result, BIG_S.result, TGT_S.result, PAIR_S.result
v("arch_g_words", HARD_S.group, "words", "group size, in words")
v("arch_g", HARD_S.group, "{:.0f}", "group size")
v("arch_big_g", BIG_S.group, "{:.0f}", "doubled group size")
v("arch_hard_pct", pct(HARD_S.pass_rate), "{:.0f}", "hard pass rate, %")
v("arch_tgt_pct", pct(TGT_S.pass_rate), "{:.0f}", "targeted pass rate, %")
v("arch_hard_c", HD.contrast, "{:.2f}", "contrast at 5%")
v("arch_hard_silent", pct(HD.silent_share), "{:.0f}", "silent groups at 5%, %")
v("arch_tgt_c", TG.contrast, "{:.2f}", "contrast at 50%")
v("arch_big_c", BG.contrast, "{:.2f}", "contrast at G 16")
v("pair_g", PAIR_S.group, "{:.0f}", "small group")
v("pair_p", PAIR_S.pass_rate, "{:.2f}", "small group's pass rate")
v("pair_c_pct", pct(PR.contrast), "{:.0f}", "small group's contrast, %")
v("budget_tasks", HARD_S.tasks, "{:.0f}", "tasks per update")
v("budget_turns", HARD_S.turns, "{:.0f}", "turns per rollout")
v("budget_tokens", HARD_S.tokens_per_turn, "{:.0f}", "tokens per turn")
v("budget_step_m", HD.step_tokens / 10 ** 6, "{:.1f}", "tokens per step, millions")
v("budget_hard_r", HD.rollouts_per_informative_group, "{:.1f}", "rollouts per informative group, G 8, 5%")
v("budget_big_r", BG.rollouts_per_informative_group, "{:.1f}", "the same, G 16")
v("budget_tgt_r", TG.rollouts_per_informative_group, "{:.1f}", "the same, 50%")
v("budget_saving", HD.rollouts_per_informative_group / TG.rollouts_per_informative_group, "{:.1f}",
  "targeting saving, x")

print("== Our run's log probabilities (data/logprob_run.json)")
_run = json.loads((DATA / "logprob_run.json").read_text())
_runs = sorted(_run["runs"], key=lambda r: r["temperature"])
_t_lo, _t_mid, _t_hi = (r["temperature"] for r in _runs)
_tok = _runs[0]["tokens"][2]          # third sampled token at the lowest temperature
_alt = _runs[-1]["tokens"][2]         # the same position at the highest temperature
_top = dict(_tok["top"])
_doc = next(t for t in _top if t not in (_tok["token"], _alt["token"]))
_p_raw = math.exp(_tok["logprob"])
# The sampled probability of the same token at the lowest temperature,
# renormalized over the reported top 3 (the omitted tail is negligible after
# the logits are divided by that temperature).
_p_t = mlsysim.physics.agents.calc_tempered_softmax([lp for _, lp in _tok["top"]], _t_lo)[
    [t for t, _ in _tok["top"]].index(_tok["token"])]
v("lp_t_lo", _t_lo, "{:.1f}", "lowest temperature")
v("lp_t_mid", _t_mid, "{:.1f}", "middle temperature")
v("lp_t_hi", _t_hi, "{:.1f}", "highest temperature")
v("lp_return", _tok["logprob"], "{:.3f}", "reported log probability of 'return'")
v("lp_if", _alt["logprob"], "{:.3f}", "reported log probability of 'if'")
v("lp_doc", _top[_doc], "{:.3f}", "reported log probability of the docstring quote")
v("lp_p_raw", _p_raw, "{:.3f}", "reported probability")
v("lp_p_t", _p_t, "{:.3f}", "probability actually sampled")
v("lp_ratio", _p_raw / _p_t, "{:.1f}", "reported over sampled, x")
v("lp_if_pct", pct(math.exp(_alt["logprob"])), "{:.1f}", "'if' probability, %")

print("== All-pass rule (Scenarios.FlakyRaceTestAllPassRule)")
FL = FLAKY_S.result
v("ap_g", FLAKY_S.group, "{:.0f}", "group of incorrect patches")
v("ap_any", pct(FL.group_false_pass), "{:.1f}", "P(at least one false pass), %")
v("ap_trials", FLAKY_S.trials, "{:.0f}", "trials")
v("ap_trials_words", FLAKY_S.trials, "words", "trials, in words")
v("ap_bad", FLAKY_S.incorrect_pass, "{:.2f}", "incorrect patch, per-trial pass")
v("ap_good", FLAKY_S.correct_pass, "{:.2f}", "correct patch, per-trial pass")
v("ap_bad_acc", pct(FL.incorrect_accepted), "{:.1f}", "incorrect accepted, %")
v("ap_good_rej", pct(FL.correct_rejected), "{:.2f}", "correct rejected, %")
v("ap_e_bad", FL.expected_trials_incorrect, "{:.2f}", "expected trials, incorrect")
v("ap_e_good", FL.expected_trials_correct, "{:.2f}", "expected trials, correct")

print("== Ratio clip (Scenarios.DominantTokenRatioClip, RareTokenRatioClip; Literature.RlTrainingDynamics)")
v("clip_eps", CLIP_HI_S.epsilon, "{:.1f}", "clip width")
v("clip_hi_p", CLIP_HI_S.probability, "{:.1f}", "dominant token")
v("clip_hi_ceil", CLIP_HI_S.result.ceiling, "{:.2f}", "its ceiling")
v("clip_lo_p", CLIP_LO_S.probability, "{:.2f}", "rare token")
v("clip_lo_rise", CLIP_LO_S.result.rise, "{:.3f}", "its allowed rise")
v("clip_lo_ceil", CLIP_LO_S.result.ceiling, "{:.3f}", "its ceiling")
v("dapo_start", RLD.DapoEntropyStartAbout, "{:.1f}", "entropy at the start, about")
v("dapo_collapse_steps", RLD.DapoSymmetricClipCollapseStepsAbout, "{:,.0f}", "collapse within, steps")
v("dapo_held", RLD.DapoClipHigherEntropyAbout, "{:.1f}", "entropy held with Clip-Higher, about")

print("== Slot cycle (Scenarios.RepairRolloutSlotResets)")
SL = SLOT_S.result
v("slot_n", SLOT_S.slots, "{:.0f}", "slots")
v("slot_decisions", SLOT_S.decisions, "{:.0f}", "decisions per episode")
v("slot_model_ms", ms(SLOT_S.model_time), "{:.0f}", "model service per decision, ms")
v("slot_tool_ms", ms(SLOT_S.tool_time), "{:.0f}", "tool execution per decision, ms")
v("slot_assess_s", sec(SLOT_S.assessment_time), "{:.1f}", "assessment, s")
v("slot_cold_s", sec(SLOT_S.cold_reset), "{:.1f}", "cold reset, s")
v("slot_snap_ms", ms(SLOT_S.snapshot_reset), "{:.0f}", "snapshot reset, ms")
v("slot_work_s", sec(SL.work), "{:.1f}", "work floor, s")
v("slot_cold_cycle", sec(SL.cold_cycle), "{:.0f}", "cold cycle, s")
v("slot_cold_cycle1", sec(SL.cold_cycle), "{:.1f}", "the same, one decimal")
v("slot_snap_cycle", sec(SL.snapshot_cycle), "{:.3f}", "snapshot cycle, s")
v("slot_cold_rate", SL.cold_episodes_per_hour, "{:,.0f}", "cold capacity, episodes an hour")
v("slot_snap_rate", SL.snapshot_episodes_per_hour, "{:,.0f}", "snapshot capacity, episodes an hour")
v("slot_ratio", SL.capacity_ratio, "{:.2f}", "capacity ratio, x")
v("slot_reset_ratio", SL.reset_ratio, "{:.0f}", "reset speedup, x")
v("slot_reset_pct", pct(1 - SL.cold_work_share), "{:.0f}", "reset share of the cold cycle, %")

print("== Barrier and verifier pool (Scenarios.RolloutGroupOfFourBarrier, RepairVerifierPool)")
BR, PL = BAR_S.result, POOL_S.result
v("bar_g", BR.group, "{:.0f}", "group size")
v("bar_idle", pct(BR.idle_share), "{:.0f}", "idle share of reserved slot time, %")
v("bar_fast", sec(BR.fastest), "{:.1f}", "fastest member, s")
v("bar_slow", sec(BR.barrier), "{:.1f}", "slowest member, s")
v("pool_traj", POOL_S.trajectories, "{:.0f}", "trajectories per interval")
v("pool_interval", sec(POOL_S.interval), "{:.0f}", "interval, s")
v("pool_rate", PL.arrival_rate.to("1/s").magnitude, "{:.0f}", "arrival rate, per s")
v("pool_trials", POOL_S.trials, "{:.0f}", "trials")
v("pool_trial_s", sec(POOL_S.trial_time), "{:.0f}", "trial time, s")
v("pool_busy", PL.busy, "{:.0f}", "busy workers")
v("pool_target", pct(POOL_S.target_utilization), "{:.0f}", "target utilization, %")
v("pool_slots", PL.slots, "{:.0f}", "provisioned slots")

print("== Yue et al. 2025 (slides/vol3/17_rlvr/images/data/yue_grpo_pass_at_k.csv)")
v("yue_p1_base", YUE[0]["pass_at_1"], "{:.1f}", "pass@1, base, %")
v("yue_p1_end", YUE[-1]["pass_at_1"], "{:.1f}", "pass@1, last step, %")
v("yue_pk_base", YUE[0]["pass_at_256"], "{:.1f}", "pass@256, base, %")
v("yue_pk_end", YUE[-1]["pass_at_256"], "{:.1f}", "pass@256, last step, %")
v("yue_steps", YUE[-1]["step"], "{:.0f}", "GRPO steps")

if __name__ == "__main__":
    write(__file__)
