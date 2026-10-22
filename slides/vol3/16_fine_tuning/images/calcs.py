#!/usr/bin/env python3
"""Every number on the Volume III Lecture 16 slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations, and the chapter reads the same
entries: the recorded role shares of 2,100 public SWE-bench Verified
trajectories (SweBenchVerifiedRoleShares over Literature.SweBenchTrajectories)
and one run's recorded tokens (SweAgentClaude35SonnetRunCost), one example per
model call (RepairTrajectoryPerCallExamples), the recovery record's target
mask (RecoveryRecordTargetMask and its variants), loss weighting
(RepairTraceWithLookupWeighting, RepairTraceWithFifteenLookups and its split),
packing (EightExamplePackingBatch), the imitation bounds
(RepairPolicyImitationBounds), one preference pair (RepairPreferencePair and
its doubled-beta variant), context distillation (SupportChecklistDistillation),
adapter memory (Rank16AdaptersOnHypothetical8B and its rank-32 variant), and
the release comparison (RepairCandidateRelease), backed by
engine/solvers/fine_tuning.py. Published data are Literature entries: the
DAgger Super Tux Kart curves (ImitationSuperTuxKart) and the LoRA paper's
GPT-3 figures (TrainingRuns.LoraGpt3*); the Llama 3 8B shape is
Models.Language.Llama3_8B. The one other source is the deck's own data
(data/): real library runs written by tool_runs.py (a chat template, packed
logits, a gradient-accumulation loss, a PEFT parameter count, and the adapter
config PEFT saves). This file only selects values, converts units, and
formats them; it types no numbers. Each value is recorded with v() under the
key the deck prints as \\val{key}; running this file writes images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Scenarios = mlsysim.Scenarios
Literature = mlsysim.Literature
Models = mlsysim.Models
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

ROLE_S = Scenarios.SweBenchVerifiedRoleShares
TOK_S = Scenarios.SweAgentClaude35SonnetRunCost
SER_S = Scenarios.RepairTrajectoryPerCallExamples
MASK_S = Scenarios.RecoveryRecordTargetMask
MASK2_S = Scenarios.RecoveryRecordTargetMaskDoubledObservations
NORM_S = Scenarios.RepairTraceWithLookupWeighting
EX3_S = Scenarios.RepairTraceWithFifteenLookups
EX3_SPLIT_S = Scenarios.RepairTraceSplitIntoTwentyRecords
PACK_S = Scenarios.EightExamplePackingBatch
BOUND_S = Scenarios.RepairPolicyImitationBounds
DPO_S = Scenarios.RepairPreferencePair
DPO2_S = Scenarios.RepairPreferencePairDoubledBeta
DIST_S = Scenarios.SupportChecklistDistillation
LORA_S = Scenarios.Rank16AdaptersOnHypothetical8B
LORA32_S = Scenarios.Rank32AdaptersOnHypothetical8B
REL_S = Scenarios.RepairCandidateRelease

ROLE = ROLE_S.result
TOK = TOK_S.result
SER = SER_S.result
MASK = MASK_S.result
NORM = NORM_S.result
EX3 = EX3_S.result
EX3_SPLIT = EX3_SPLIT_S.result
PACK = PACK_S.result
DPO = DPO_S.result
DPO2 = DPO2_S.result
DIST = DIST_S.result
REL = REL_S.result
STK = Literature.ImitationSuperTuxKart
LLAMA = Models.Language.Llama3_8B
TR = Literature.TrainingRuns


def pct(x):
    return x * 100


def usd(q):
    return q.to("USD").magnitude


def gb(q):
    return q.to("GB").magnitude


def run_value(fname, pattern):
    return re.search(pattern, (DATA / fname).read_text()).group(1)


print("== Hook and synthesis: recorded role shares (Scenarios.SweBenchVerifiedRoleShares)")
_runs = [c["runs"] for c in ROLE.configurations]
v("hook_runs", ROLE.runs, "{:,.0f}", "public trajectories")
v("hook_cfg_runs", max(_runs), "{:.0f}", "runs per full configuration")
v("hook_cfg_full_words", _runs.count(max(_runs)), "words", "configurations with that many runs")
v("hook_cfg_small_runs", min(_runs), "{:.0f}", "runs in the sampled configuration")
v("hook_cfg_small_words", _runs.count(min(_runs)), "words", "configurations sampled")
v("hook_model_lo", pct(ROLE.model_low), "{:.0f}", "model output, lowest configuration, %")
v("hook_model_hi", pct(ROLE.model_high), "{:.0f}", "model output, highest configuration, %")
v("hook_obs_lo", pct(ROLE.observation_low), "{:.0f}", "observations, lowest configuration, %")
v("hook_obs_hi", pct(ROLE.observation_high), "{:.0f}", "observations, highest configuration, %")

print("== Chat template (data/template_render.txt, our run)")
_t_asst = int(run_value("template_render.txt", r"assistant turn (\d+)"))
_t_all = int(run_value("template_render.txt", r"total (\d+)"))
v("tmpl_asst", _t_asst, "{:.0f}", "assistant-turn tokens")
v("tmpl_all", _t_all, "{:.0f}", "tokens, from the user turn on")
v("tmpl_asst_pct", pct(_t_asst / _t_all), "{:.1f}", "model output share, %")

print("== One example per call (Scenarios.RepairTrajectoryPerCallExamples)")
v("ser_turns", SER_S.turns, "{:.0f}", "stipulated turns")
v("ser_exp", SER.expansion, "{:.1f}", "stored tokens over one transcript, x")

print("== Recorded tokens of one run (Scenarios.SweAgentClaude35SonnetRunCost)")
v("tok_runs", TOK.tasks, "{:.0f}", "SWE-agent runs")
v("tok_calls", TOK.median_calls, "{:.0f}", "median model calls")
v("tok_in", TOK.median_input_tokens, "{:,.0f}", "median input tokens, summed over calls")
v("tok_out", TOK.median_output_tokens, "{:,.0f}", "median output tokens")
v("tok_ratio", TOK.median_input_output_ratio, "{:.0f}", "median input/output ratio")
v("tok_out_pct", pct(TOK.output_token_share), "{:.0f}", "output share of all tokens, pooled, %")

print("== Target mask (Scenarios.RecoveryRecordTargetMask)")
v("mask_stored", MASK.stored, "{:,.0f}", "stored tokens")
v("mask_pred", MASK.predicted, "{:,.0f}", "prediction terms")
v("mask_obs", MASK_S.observation, "{:,.0f}", "observation tokens")
v("mask_obs_pct", pct(MASK.observation_share), "{:.1f}", "observation share, %")
v("mask_resp", MASK_S.response, "{:.0f}", "response tokens")
v("mask_refused", MASK_S.unselected_response, "{:.0f}", "refused-proposal tokens")
v("mask_rat", MASK_S.rationale, "{:,.0f}", "rationale tokens")
v("mask_tgt", MASK.targets_without_rationale, "{:.0f}", "targets, rationale as context")
v("mask_tgt_pct", pct(MASK.share_without_rationale), "{:.1f}", "the same, % of stored")
v("mask_tgt_rat", MASK.targets_with_rationale, "{:,.0f}", "targets, rationale selected")
v("mask_tgt_rat_pct", pct(MASK.share_with_rationale), "{:.1f}", "the same, % of stored")
v("mask_dil", MASK.dilution_without_rationale, "{:.1f}", "dilution of each target, x")
v("mask_dil_rat", MASK.dilution_with_rationale, "{:.1f}", "dilution with rationale selected, x")
v("mask_dil_double", MASK2_S.result.dilution_without_rationale, "{:.1f}", "dilution, observations doubled, x")
v("mask_refused_pct", pct(MASK.unselected_share_of_responses), "{:.1f}", "refused share of response weight, %")

print("== Gradient accumulation (data/accum_test.txt, our run)")
_acc_targets = re.search(r"selected targets: (\d+) and (\d+)", (DATA / "accum_test.txt").read_text())
_acc_losses = re.search(r"mean losses:\s+(\S+)\s+(\S+)", (DATA / "accum_test.txt").read_text())
_mom = float(run_value("accum_test.txt", r"mean of micro-batch means:\s+(\S+)"))
_glob = float(run_value("accum_test.txt", r"num_items_in_batch=\d+:\s+(\S+)"))
v("acc_n1", int(_acc_targets.groups()[0]), "{:,.0f}", "targets, micro-batch 1")
v("acc_n2", int(_acc_targets.groups()[1]), "{:.0f}", "targets, micro-batch 2")
v("acc_l1", float(_acc_losses.groups()[0]), "{:.3f}", "mean loss, micro-batch 1")
v("acc_l2", float(_acc_losses.groups()[1]), "{:.3f}", "mean loss, micro-batch 2")
v("acc_mom", _mom, "{:.3f}", "mean of the two means")
v("acc_glob", _glob, "{:.3f}", "global token average")
v("acc_ratio", _mom / _glob, "{:.1f}", "mean of means over global average, x")

print("== Normalization (Scenarios.RepairTraceWithLookupWeighting)")
v("norm_short", NORM_S.other_targets, "{:.0f}", "lookup targets")
v("norm_long", NORM_S.focus_targets, "{:,.0f}", "repair-trace targets")
v("norm_w_short", pct(NORM.other_token_share), "{:.2f}", "lookup weight, token averaging, %")
v("norm_w_long", pct(NORM.focus_token_share), "{:.2f}", "trace weight, token averaging, %")
v("norm_ratio", NORM.focus_over_other, "{:.0f}", "weight ratio, x")
v("norm_ex", pct(NORM.focus_example_share), "{:.0f}", "each example, example averaging, %")

print("== Exercise 3 (Scenarios.RepairTraceWithFifteenLookups and its split)")
v("ex3_long", EX3_S.focus_targets, "{:,.0f}", "trace targets")
v("ex3_n_short", EX3_S.other_examples, "{:.0f}", "file lookups")
v("ex3_short", EX3_S.other_targets, "{:.0f}", "targets per lookup")
v("ex3_total", EX3.total_targets, "{:,.0f}", "targets in the batch")
v("ex3_examples", EX3.examples, "{:.0f}", "examples, unsplit")
v("ex3_split", EX3_SPLIT_S.focus_pieces, "{:.0f}", "per-call records")
v("ex3_piece", EX3_S.focus_targets / EX3_SPLIT_S.focus_pieces, "{:.0f}", "targets per record")
v("ex3_examples_split", EX3_SPLIT.examples, "{:.0f}", "examples, split")
v("ex3_tok", pct(EX3.focus_token_share), "{:.1f}", "trace share, token averaging, %")
v("ex3_ex", pct(EX3.focus_example_share), "{:.2f}", "trace share, example averaging, %")
v("ex3_ex_round", pct(EX3.focus_example_share), "{:.0f}", "the same, whole percent")
v("ex3_ex_split", pct(EX3_SPLIT.focus_example_share), "{:.1f}", "trace share after the split, %")
v("ex3_ex_split_round", pct(EX3_SPLIT.focus_example_share), "{:.0f}", "the same, whole percent")
v("ex3_rise_words", round(EX3_SPLIT.focus_example_share / EX3.focus_example_share), "words",
  "rise of the example-averaged share, about")

print("== Packing (Scenarios.EightExamplePackingBatch)")
v("pack_n_words", PACK.examples, "words", "examples")
v("pack_rest_words", PACK.examples - 1, "words", "examples in the second buffer")
v("pack_min", min(PACK_S.lengths), "{:.0f}", "shortest example, tokens")
v("pack_max", PACK.longest, "{:,.0f}", "longest example and buffer, tokens")
v("pack_stored", PACK.stored, "{:,.0f}", "stored tokens")
v("pack_padded", PACK.padded_positions, "{:,.0f}", "padded positions")
v("pack_pad_occ", pct(PACK.padded_occupancy), "{:.1f}", "padded occupancy, %")
v("pack_occ", pct(PACK.packed_occupancy), "{:.1f}", "packed occupancy, %")
v("pack_fewer", PACK.position_ratio, "{:.0f}", "fewer positions, x")

print("== Packed-example isolation (data/packing_test.txt, our run)")
_causal = float(run_value("packing_test.txt", r"ordinary causal mask:\s+(\S+)"))
_block = float(run_value("packing_test.txt", r"pos reset:\s+(\S+)"))
_offset = float(run_value("packing_test.txt", r"pos offset:\s+(\S+)"))
_exp = math.floor(math.log10(_offset))
v("blk_causal", _causal, "{:.2f}", "max logit difference, ordinary causal mask")
v("blk_block", _block, "{:.0f}", "max logit difference, block-diagonal, positions reset")
v("blk_off_mant", _offset / 10 ** _exp, "{:.0f}", "positions kept: mantissa")
v("blk_off_exp", _exp, "{:.0f}", "positions kept: power of ten")
v("blk_layers", int(run_value("packing_test.txt", r"(\d+) layers")), "{:.0f}", "layers of the random model")

print("== Imitation bounds (Scenarios.RepairPolicyImitationBounds)")
_h = BOUND_S.horizon_steps
v("eb_h", _h, "{:.0f}", "horizon, steps")
v("eb_h_sq", _h * _h, "{:.0f}", "H squared")
v("eb_eps", BOUND_S.per_step_error, "{:.2f}", "disagreement rate")
v("eb_eps_pct", pct(BOUND_S.per_step_error), "{:.0f}", "the same, %")
v("eb_u", BOUND_S.recoverability, "{:.0f}", "deviation disadvantage u")
v("eb_ref", BOUND_S.cloning_capped, "{:.0f}", "bound on reference states")
v("eb_lrn", BOUND_S.aggregation_capped, "{:.1f}", "bound on learner states")
v("eb_gap", BOUND_S.capped_gap, "{:.0f}", "gap, x")
v("eb_need_pct", pct(BOUND_S.matching_error), "{:.3f}", "reference error that matches, %")
v("eb_sat", BOUND_S.saturation_horizon, "{:.0f}", "saturation horizon, steps")

print("== Super Tux Kart (Literature.ImitationSuperTuxKart, read from Ross et al. Figure 2)")
_sup = sorted(STK.where(method="Supervised"), key=lambda r: r["iteration"])
_dag = sorted(STK.where(method="DAgger"), key=lambda r: r["iteration"])
v("stk_states", STK.entries[0]["states_per_iteration"], "{:,.0f}", "states collected per iteration")
v("stk_first_it", _sup[0]["iteration"], "{:.0f}", "first iteration")
v("stk_last_it", _sup[-1]["iteration"], "{:.0f}", "last iteration")
v("stk_sup_first", _sup[0]["falls_per_lap"], "{:.0f}", "supervised falls per lap, first iteration")
v("stk_sup_last", _sup[-1]["falls_per_lap"], "{:.0f}", "supervised falls per lap, last iteration")
v("stk_dag_low_it", _dag[1]["iteration"], "{:.0f}", "second recorded iteration: DAgger almost never falls")
v("stk_dag_zero_it", next(r["iteration"] for r in _dag if r["falls_per_lap"] == 0), "{:.0f}",
  "first iteration with no DAgger falls")

print("== Preference pair (Scenarios.RepairPreferencePair and its doubled beta)")
v("dpo_pol_w", DPO_S.policy_preferred, "{:.0f}", "policy, preferred")
v("dpo_ref_w", DPO_S.reference_preferred, "{:.0f}", "reference, preferred")
v("dpo_pol_l", DPO_S.policy_dispreferred, "{:.0f}", "policy, dispreferred")
v("dpo_ref_l", DPO_S.reference_dispreferred, "{:.0f}", "reference, dispreferred")
v("dpo_zw", DPO.score_preferred, "{:.0f}", "z_w")
v("dpo_zl", DPO.score_dispreferred, "{:.0f}", "z_l")
v("dpo_beta", DPO_S.beta, "{:.1f}", "beta")
v("dpo_beta2", DPO2_S.beta, "{:.1f}", "doubled beta")
v("dpo_margin", DPO.margin, "{:.1f}", "margin")
v("dpo_margin2", DPO2.margin, "{:.1f}", "margin, doubled beta")
v("dpo_score", DPO.preference_score, "{:.3f}", "logistic score")
v("dpo_score_pct", pct(DPO.preference_score), "{:.1f}", "the same, %")
v("dpo_score2", DPO2.preference_score, "{:.3f}", "logistic score, doubled beta")
v("dpo_loss", DPO.loss, "{:.3f}", "pair loss")
v("dpo_loss0", DPO.initial_loss, "{:.3f}", "pair loss at a zero margin")
v("dpo_loss2", DPO2.loss, "{:.3f}", "pair loss, doubled beta")
v("dpo_sat", DPO.saturation_factor, "{:.3f}", "saturation factor")
v("dpo_coef", DPO.gradient_coefficient, "{:.4f}", "gradient coefficient")
v("dpo_coef2", DPO2.gradient_coefficient, "{:.4f}", "gradient coefficient, doubled beta")

print("== Context distillation (Scenarios.SupportChecklistDistillation)")
v("dist_prefix", DIST_S.prefix_tokens, "{:,.0f}", "checklist tokens")
v("dist_turns", DIST_S.turns_per_session, "{:.0f}", "turns per session")
v("dist_sessions_m", DIST_S.sessions_per_month / 10 ** 6, "{:.0f}", "sessions per month, millions")
v("dist_tokens_b", DIST.monthly_prefix_tokens / 10 ** 9, "{:.0f}", "checklist tokens per month, billions")
v("dist_price", DIST_S.input_price.to("USD / Mtoken").magnitude, "{:.0f}", "input price, $ per million")
v("dist_unc", usd(DIST.uncached_spend), "{:,.0f}", "spend removed, uncached, $")
v("dist_cac", usd(DIST.cached_spend), "{:,.0f}", "spend removed, cached, $")
v("dist_redistill", usd(DIST_S.redistill_cost), "{:,.0f}", "redistillation, $")
v("dist_net_unc", usd(DIST.net_uncached), "{:,.0f}", "net saving, uncached, $ a month")
v("dist_net_cac", -usd(DIST.net_cached), "{:,.0f}", "net loss, cached, $ a month")
v("dist_breakeven", DIST.break_even_revisions, "{:.1f}", "revisions a month that cancel the saving")

print("== Low-rank adapters (Scenarios.Rank16AdaptersOnHypothetical8B)")
L = LORA_S
v("lora_layers", L.workload.layers, "{:.0f}", "layers")
v("lora_mats", L.adapted_matrices, "{:.0f}", "adapted matrices per layer")
v("lora_mats_words", L.adapted_matrices, "words", "the same, in words")
v("lora_rank", L.adapter_rank, "{:.0f}", "rank")
v("lora_din", L.adapter_in_dim, "{:,.0f}", "input width")
v("lora_dout", L.adapter_out_dim, "{:,.0f}", "output width")
v("lora_params", L.adapter_count, "{:,.0f}", "adapter parameters")
v("lora_frac_pct", pct(L.adapter_fraction), "{:.2f}", "share of the base, %")
v("lora_frac_pct1", pct(L.adapter_fraction), "{:.1f}", "the same, one decimal")
v("lora_base_b", L.workload.parameters.to("count").magnitude / 10 ** 9, "{:.0f}", "base parameters, billions")
v("lora_b_trained", L.bytes_per_trained_parameter.to("byte").magnitude, "{:.0f}", "bytes per trained parameter")
v("lora_b_frozen", L.bytes_per_value.to("byte").magnitude, "{:.0f}", "bytes per frozen parameter")
v("lora_b_master", L.master_bytes_per_parameter.to("byte").magnitude, "{:.0f}", "master copy, bytes")
v("lora_b_moment", mlsysim.core.units.BYTES_FP32.to("byte").magnitude, "{:.0f}",
  "one Adam moment (an FP32 value), bytes")
v("lora_full_gb", gb(L.full_state), "{:.0f}", "full fine-tuning state, GB")
v("lora_frozen_gb", gb(L.weights), "{:.0f}", "frozen base, GB")
v("lora_ad_mb", L.adapter_state.to("MB").magnitude, "{:.2f}", "adapter state, MB")
v("lora_ad_mb0", L.adapter_state.to("MB").magnitude, "{:.0f}", "the same, whole MB")
v("lora_gb", gb(L.total), "{:.2f}", "adapter run state, GB")
v("lora_keep_pct", pct(L.adapter_kept_share), "{:.1f}", "kept share of full state, %")
v("lora_keep_pct0", pct(L.adapter_kept_share), "{:.0f}", "the same, whole percent")
v("lora_red", L.adapter_reduction, "{:.1f}", "reduction, x")
v("lora_frozen_share", pct(L.frozen_share), "{:.1f}", "frozen share of the adapter run, %")
v("lora_bound", L.adapter_reduction_ceiling, "{:.0f}", "reduction ceiling, x")
v("lora_rank_add_mb", (LORA32_S.adapter_state - L.adapter_state).to("MB").magnitude, "{:.0f}",
  "doubling the rank adds, MB")

print("== Llama 3 8B projections (Models.Language.Llama3_8B)")
v("llama_d", LLAMA.hidden_dim, "{:,.0f}", "hidden width")
v("llama_kv_d", LLAMA.kv_heads * (LLAMA.hidden_dim // LLAMA.heads), "{:,.0f}", "key and value projection width")

print("== PEFT on the Llama 3 8B architecture (data/lora_count.txt, our run)")
_peft = (DATA / "lora_count.txt").read_text()
v("peft_r", int(re.search(r"r=(\d+)", _peft).group(1)), "{:.0f}", "rank")
v("peft_alpha", int(re.search(r"lora_alpha=(\d+)", _peft).group(1)), "{:.0f}", "lora_alpha")
_tr = int(run_value("lora_count.txt", r"trainable params: ([\d,]+)").replace(",", ""))
_all = int(run_value("lora_count.txt", r"all params: ([\d,]+)").replace(",", ""))
v("peft_tr", _tr, "{:,.0f}", "trainable parameters")
v("peft_all", _all, "{:,.0f}", "all parameters, adapters included")
v("peft_all_b", _all / 10 ** 9, "{:.2f}", "the same, billions")
v("peft_pct", float(run_value("lora_count.txt", r"trainable%: (\S+)")), "{:.4f}", "trainable, %")
_cfg = json.loads((DATA / "adapter_config.json").read_text())
v("peft_cfg_keys", len(_cfg), "{:.0f}", "keys in adapter_config.json")

print("== LoRA paper, GPT-3 175B (Literature.TrainingRuns.LoraGpt3*)")
_q = mlsysim.core.units.Q_
_vram_full = _q(TR.LoraGpt3TrainingMemoryFullGB, "GB")
_vram_lora = _q(TR.LoraGpt3TrainingMemoryLoraGB, "GB")
_ckpt_full = _q(TR.LoraGpt3CheckpointFullGB, "GB")
_ckpt_lora = _q(TR.LoraGpt3CheckpointLoraMB, "MB")
_n = TR.LoraGpt3AdaptedModels
v("hu_vram_full_tb", _vram_full.to("TB").magnitude, "{:.1f}", "full fine-tuning VRAM, TB")
v("hu_vram_lora_gb", gb(_vram_lora), "{:.0f}", "LoRA VRAM, GB")
v("hu_vram_red", (_vram_full / _vram_lora).to("").magnitude, "{:.1f}", "VRAM reduction, x")
v("hu_n", _n, "{:.0f}", "adapted models")
v("hu_hundred_lora_gb", gb(_ckpt_full + _n * _ckpt_lora), "{:.0f}", "base plus adapters, GB")
v("hu_hundred_full_tb", (_n * _ckpt_full).to("TB").magnitude, "{:.0f}", "full copies, TB")

print("== Release (Scenarios.RepairCandidateRelease)")
R = REL_S
v("rel_trials", R.trials, "{:.0f}", "trials per arm")
v("rel_p_b", pct(R.baseline_acceptance), "{:.0f}", "baseline acceptance, %")
v("rel_p_c", pct(R.candidate_acceptance), "{:.0f}", "candidate acceptance, %")
v("rel_p_b_dec", R.baseline_acceptance, "{:.2f}", "baseline acceptance")
v("rel_p_c_dec", R.candidate_acceptance, "{:.2f}", "candidate acceptance")
v("rel_c_b", usd(R.baseline_attempt_cost), "{:.2f}", "baseline attempt cost, $")
v("rel_c_c", usd(R.candidate_attempt_cost), "{:.2f}", "candidate attempt cost, $")
v("rel_ce_b", usd(REL.baseline_cost_per_accepted), "{:.2f}", "baseline C_eff, $")
v("rel_ce_c", usd(REL.candidate_cost_per_accepted), "{:.2f}", "candidate C_eff, $")
v("rel_rise", pct(REL.cost_rise), "{:.0f}", "C_eff rise, %")
v("rel_breakeven", usd(REL.break_even_attempt_cost), "{:.2f}", "break-even attempt cost, $")
v("rel_gap_pts", pct(REL.gap), "{:.0f}", "acceptance gain, points")
v("rel_se", REL.gap_se, "{:.4f}", "standard error of the difference")
v("rel_z", REL.gap_z, "{:.2f}", "difference in standard errors")
v("rel_sfa_fam_b", R.baseline_sfa_familiar, "{:.2f}", "SFA, familiar, baseline")
v("rel_sfa_fam_c", R.candidate_sfa_familiar, "{:.2f}", "SFA, familiar, candidate")
v("rel_sfa_ren_b", R.baseline_sfa_renamed, "{:.2f}", "SFA, renamed, baseline")
v("rel_sfa_ren_c", R.candidate_sfa_renamed, "{:.2f}", "SFA, renamed, candidate")
v("rel_pre_b", R.baseline_pre_renamed, "{:.2f}", "PRE, renamed, baseline")
v("rel_pre_c", R.candidate_pre_renamed, "{:.2f}", "PRE, renamed, candidate")
v("rel_pre_words", REL.recall_error_ratio, "words", "PRE rise, fold")

if __name__ == "__main__":
    write(__file__)
