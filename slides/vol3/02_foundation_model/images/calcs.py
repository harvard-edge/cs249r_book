#!/usr/bin/env python3
"""Every number on the Volume III Lecture 2 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (registry, Literature, the coding and support agent
profiles) and the calculations (Scenarios.*, each backed by a solver in
engine/solvers/model_call.py); the chapter reads the same scenarios. The one
other source is the deck's own measurement, data/model_runs.json, written by
run_model.py (Qwen2.5-Coder-0.5B-Instruct on a laptop CPU); this file reads it
as recorded. This file only selects values, converts units, and formats them;
it types no numbers. Each value is recorded with v() under the key the deck
prints as \\val{key}; running this file writes images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Literature, Scenarios = mlsysim.Literature, mlsysim.Scenarios
RUN = json.loads((Path(__file__).resolve().parent / "data" / "model_runs.json").read_text())

FLOORS = Scenarios.Llama3_70B_Int8DecodeH100
R30, R60 = Scenarios.CodingAgentResend30Turns, Scenarios.CodingAgentResend60Turns
CALL1 = Scenarios.CodingAgentFirstTurnCall
FULL, DIFF = Scenarios.CodingAgentWholeFileEdit, Scenarios.CodingAgentLocalizedEdit
CANCEL = Scenarios.CanceledStreamDecodeSavings
ADMIT, HALF = Scenarios.MultimodalContextAdmission, Scenarios.MultimodalContextAdmissionHalfReserve
SAMPLE = Scenarios.TemperatureNucleusExample
CHECK, CHECK_HALF = Scenarios.PatchCheckerPosterior, Scenarios.PatchCheckerHalvedFalseAccept
HORIZON, COST = Scenarios.HorizonCompounding30Steps, Scenarios.CodingAgentCostPerAccepted
TRACE = Scenarios.LlmCodeTraceTokenMix
INCIDENT, SELF = Literature.ServingIncidents, Literature.SelfCorrection


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


print("== Hook and synthesis (Anthropic 2025 postmortem)")
v("pm_initial_pct", 100 * INCIDENT.Anthropic2025MisroutedShareInitial, "{:.1f}", "misrouted at first, %")
v("pm_worst_pct", 100 * INCIDENT.Anthropic2025MisroutedShareWorstHour, "{:.0f}", "misrouted at the worst hour, %")
v("pm_users_pct", 100 * INCIDENT.Anthropic2025AffectedUserShare, "{:.0f}", "coding-tool users affected, %")
v("pm_topk_n", INCIDENT.Anthropic2025TopKProbeLength, "{:,.0f}", "top-k reproduction array length")
v("pm_topk_fail", INCIDENT.Anthropic2025TopKFirstFailingIndex, "{:,.0f}", "first failing index")

print("== Recall: horizon compounding and cost per accepted task")
v("rec_h", HORIZON.horizon, "{:.0f}", "repair horizon, steps (harness step budget)")
v("rec_step", HORIZON.step_success, "{:.2f}", "per-step success")
v("rec_step_pct", 100 * HORIZON.step_success, "{:.0f}", "per-step success, %")
v("rec_success_pct", 100 * HORIZON.success, "{:.1f}", "task success, %")
v("rec_attempt", m(COST.attempt_cost, "USD"), "{:.2f}", "cost per attempt, $")
v("rec_acc", COST.acceptance, "{:.2f}", "acceptance probability")
v("rec_acc_pct", 100 * COST.acceptance, "{:.0f}", "acceptance probability, %")
v("rec_ceff", m(COST.cost_per_accepted, "USD"), "{:.2f}", "cost per accepted task, $")

print("== The real call (run_model.py)")
vis_sys, vis_user = RUN["visible_tokens"]["system"], RUN["visible_tokens"]["user"]
vis, wire = vis_sys + vis_user, RUN["wire_tokens"]
v("run_done_tokens", RUN["completed_call"]["new_tokens"], "{:.0f}", "completed call, new tokens")
v("run_done_max", RUN["completed_call"]["max_new_tokens"], "{:.0f}", "completed call, max_new_tokens")
v("run_cut_tokens", RUN["truncated_call"]["new_tokens"], "{:.0f}", "truncated call, new tokens")
v("run_cut_max", RUN["truncated_call"]["max_new_tokens"], "{:.0f}", "truncated call, max_new_tokens")
v("wire_tokens", wire, "{:.0f}", "serialized request tokens")
v("visible_tokens", vis, "{:.0f}", "visible message tokens")
v("visible_sys", vis_sys, "{:.0f}", "system message tokens")
v("visible_user", vis_user, "{:.0f}", "user message tokens")
v("framing_tokens", wire - vis, "{:.0f}", "framing tokens")
v("wire_ratio", wire / vis, "{:.1f}", "wire / visible")
v("vocab", RUN["vocab_size"], "{:,.0f}", "vocabulary size")

print("== Tokenizer (cl100k_base, run_model.py)")
flat, ind, path = RUN["cl100k_flat"]["ids"], RUN["cl100k_indented"]["ids"], RUN["cl100k_path"]["ids"]
v("tok_flat_n", len(flat), "{:.0f}", "flush line tokens")
v("tok_flat_id0", flat[0], "{:.0f}", "id of 'def'")
v("tok_ind_n", len(ind), "{:.0f}", "indented line tokens")
v("tok_ind_id0", ind[0], "{:.0f}", "id of three spaces")
v("tok_ind_id1", ind[1], "{:.0f}", "id of ' def'")
v("tok_path_n", len(path), "{:.0f}", "src/pool.py tokens")

print("== Admission (multimodal admission scenario)")
a, h = ADMIT.result, HALF.result
v("adm_window", ADMIT.window, "{:,.0f}", "context window, tokens")
v("adm_reserve", ADMIT.output_reserve, "{:,.0f}", "reserved output, tokens")
v("adm_text", ADMIT.text_tokens, "{:,.0f}", "text prompt, tokens")
v("adm_image", ADMIT.image_tokens, "{:,.0f}", "image charge, tokens")
v("adm_ceiling", a.input_ceiling, "{:,.0f}", "input ceiling")
v("adm_headroom", a.headroom, "{:,.0f}", "room left")
v("adm_total", a.total_demand, "{:,.0f}", "demand with image")
v("adm_excess", a.excess, "{:,.0f}", "excess with image")
v("adm_half_reserve", HALF.output_reserve, "{:,.0f}", "halved reserve")
v("adm_half_total", h.total_demand, "{:,.0f}", "demand with image, reserve halved")
v("adm_half_excess", h.excess, "{:,.0f}", "excess with image, reserve halved")

print("== Real next-token distribution (run_model.py)")
pl = RUN["path_logits"]
i_path, i_c, i_src = pl["tokens"].index("path"), pl["tokens"].index("C"), pl["tokens"].index("src")
v("p_path_t1", 100 * pl["p_T1"][i_path], "{:.1f}", "'path' at T = 1, %")
v("p_c_t1", 100 * pl["p_T1"][i_c], "{:.1f}", "'C' at T = 1, %")
v("p_src_t1", 100 * pl["p_T1"][i_src], "{:.1f}", "'src' at T = 1, %")
v("p_path_t05", 100 * pl["p_T05"][i_path], "{:.1f}", "'path' at T = 0.5, %")
v("p_src_t05", 100 * pl["p_T05"][i_src], "{:.1f}", "'src' at T = 0.5, %")
v("run_nucleus", pl["nucleus_090_size"], "{:.0f}", "top-p 0.90 nucleus size at T = 1, tokens")
v("run_top_n", len(pl["tokens"]), "{:.0f}", "candidates shown")

print("== Napkin Math 2.1 (temperature and nucleus scenario)")
s = SAMPLE.result
for i, z in enumerate(SAMPLE.logits):
    v(f"nm_z{i + 1}", z, "{:g}", f"logit v{i + 1}")
v("nm_t_base", SAMPLE.base_temperature, "{:g}", "base temperature")
v("nm_t_sharp", SAMPLE.sharp_temperature, "{:g}", "sharpened temperature")
v("nm_top_p", SAMPLE.top_p, "{:.2f}", "top-p")
v("nm_p1_pct", 100 * s.base_probabilities[0], "{:.1f}", "p(v1) at T = 1, %")
v("nm_p1", s.base_probabilities[0], "{:.4f}", "p(v1) at T = 1")
v("nm_sharp_pct", 100 * s.sharp_probabilities[0], "{:.1f}", "p(v1) at T = 0.5, %")
v("nm_sharp_round", 100 * s.sharp_probabilities[0], "{:.0f}", "the same, whole percent (vote answer)")
v("nm_tail1_pct", 100 * s.base_tail, "{:.1f}", "tail at T = 1, %")
v("nm_tail05_pct", 100 * s.sharp_tail, "{:.1f}", "tail at T = 0.5, %")
v("nm_tail_ratio", s.tail_ratio, "{:.1f}", "tail reduction")
v("nm_nuc_mass", s.nucleus_mass, "{:.4f}", "top-2 cumulative")
v("nm_nuc1_pct", 100 * s.nucleus_probabilities[0], "{:.1f}", "renormalized v1, %")
v("nm_nuc2_pct", 100 * s.nucleus_probabilities[1], "{:.1f}", "renormalized v2, %")
v("nm_nuc1", s.nucleus_probabilities[0], "{:.4f}", "renormalized v1")
v("nm_nuc2", s.nucleus_probabilities[1], "{:.4f}", "renormalized v2")

print("== Batch-dependent logits (run_model.py)")
bl, vocab = RUN["batch_logits"], RUN["vocab_size"]
d32 = bl["float32"]["max_abs_diff"]
e32 = math.floor(math.log10(d32))
v("batch_other", RUN["batch_other_tokens"], "{:.0f}", "other request tokens")
v("batch_pad", bl["float32"]["pad_tokens_added"], "{:.0f}", "padding tokens added")
v("batch_fp32_changed", bl["float32"]["n_logits_changed"], "{:,.0f}", "fp32 logits changed")
v("batch_bf16_changed", bl["bfloat16"]["n_logits_changed"], "{:,.0f}", "bf16 logits changed")
v("batch_fp32_pct", 100 * bl["float32"]["n_logits_changed"] / vocab, "{:.1f}", "fp32 share changed, %")
v("batch_fp32_mant", d32 / 10 ** e32, "{:.1f}", "fp32 max change, mantissa")
v("batch_fp32_exp", e32, "{:.0f}", "fp32 max change, power of ten")
v("batch_bf16_max", bl["bfloat16"]["max_abs_diff"], "{:.2f}", "bf16 max change")

print("== Streaming cancel (canceled-stream scenario)")
v("cancel_cap", CANCEL.output_allowance, "{:,.0f}", "output allowance")
v("cancel_step", CANCEL.reject_step, "{:.0f}", "reject step")
v("cancel_skipped", CANCEL.skipped_steps, "{:,.0f}", "decode steps skipped")
v("cancel_s", m(CANCEL.decode_time_avoided, "s"), "{:.1f}", "decode time avoided, s")

print("== Self-review (Huang et al. 2024, GPT-4)")
for key, name in (("gsm", "Gsm8k"), ("csqa", "CommonSenseQa"), ("hotpot", "HotpotQa")):
    v(f"huang_{key}_std", getattr(SELF, f"Gpt4{name}Standard"), "{:.1f}", f"{name} one answer, %")
    v(f"huang_{key}_sc", getattr(SELF, f"Gpt4{name}SelfCorrectRound2"), "{:.1f}", f"{name} two rounds, %")
    v(f"huang_{key}_orc", getattr(SELF, f"Gpt4{name}Oracle"), "{:.1f}", f"{name} oracle-stopped, %")

print("== Residual defects (patch-checker scenario)")
c, ch = CHECK.result, CHECK_HALF.result
v("chk_defect_pct", 100 * CHECK.defect_rate, "{:.0f}", "defective share, %")
v("chk_defect", CHECK.defect_rate, "{:.2f}", "defective share")
v("chk_valid_share", 1 - CHECK.defect_rate, "{:.2f}", "valid share")
v("chk_fa", CHECK.false_accept_rate, "{:.2f}", "P(PASS | defective)")
v("chk_va", CHECK.valid_accept_rate, "{:.2f}", "P(PASS | valid)")
v("chk_post_pct2", 100 * c.posterior, "{:.2f}", "residual defect among passes, %")
v("chk_post_pct", 100 * c.posterior, "{:.1f}", "the same, one decimal")
v("chk_catch_pct", 100 * c.catch_rate, "{:.0f}", "defective patches rejected, %")
v("chk_vreject_pct", 100 * c.valid_reject_rate, "{:.0f}", "valid patches rejected, %")
v("chk_fa_half", CHECK_HALF.false_accept_rate, "{:.3f}", "halved false-accept rate")
v("chk_half_pct", 100 * ch.posterior, "{:.1f}", "residual with false accepts halved, %")
v("chk_joint_pct", 100 * c.defective_pass, "{:.1f}", "joint defective-and-pass, %")

print("== Per-token floors (8-bit floors scenario)")
f = FLOORS.result
v("fl_params_b", m(FLOORS.workload.parameters, "Gparam"), "{:.1f}", "parameters, billions")
v("fl_weights_gb", m(f.weight_bytes, "GB"), "{:.1f}", "weight bytes, GB")
v("fl_bytes_per_param", m(f.weight_bytes, "byte") / m(FLOORS.workload.parameters, "param"), "{:.0f}",
  "bytes per parameter")
v("fl_peak_tflops", m(FLOORS.hardware.compute.precision_flops[FLOORS.peak_precision], "TFLOP/s"), "{:,.0f}",
  "FP8 peak, TFLOP/s")
v("fl_bw_tbs", m(FLOORS.hardware.memory.bandwidth, "TB/s"), "{:.2f}", "HBM bandwidth, TB/s")
v("fl_prefill_us", m(f.compute_time, "us"), "{:.2f}", "prefill floor, us/token")
v("fl_decode_ms", m(f.memory_time, "ms"), "{:.3f}", "decode floor, ms/token")
v("fl_ratio", f.bound_ratio, "{:.0f}", "decode / prefill")
t1 = CALL1.result
v("t1_prompt", CALL1.prompt_tokens, "{:,.0f}", "first turn, prompt tokens")
v("t1_output", CALL1.output_tokens, "{:.0f}", "first turn, output tokens")
v("t1_prefill_s", m(t1.prefill_time, "s"), "{:.3f}", "first turn prefill, s")
v("t1_decode_s", m(t1.decode_time, "s"), "{:.3f}", "first turn decode, s")
v("t1_tok_pct", 100 * t1.output_token_share, "{:.1f}", "output share of tokens, %")
v("t1_time_pct", 100 * t1.output_time_share, "{:.1f}", "output share of time, %")

print("== Measured on a laptop (run_model.py)")
tm = RUN["timing"]
pre_tok = tm["prefill_s_median"] / tm["prompt_tokens"]
v("lap_prompt", tm["prompt_tokens"], "{:,.0f}", "prompt tokens")
v("lap_prefill_s", tm["prefill_s_median"], "{:.2f}", "prefill median, s")
v("lap_prefill_ms", pre_tok * 10 ** 3, "{:.1f}", "prefill per token, ms")
v("lap_prefill_ms2", pre_tok * 10 ** 3, "{:.2f}", "the same, two decimals (figure)")
v("lap_decode_ms", tm["decode_s_per_token_median"] * 10 ** 3, "{:.1f}", "decode per token, ms")
v("lap_ratio", tm["decode_s_per_token_median"] / pre_tok, "{:.0f}", "decode / prefill")
v("lap_ratio_lo", min(tm["decode_s_per_token_runs"]) / (max(tm["prefill_s_runs"]) / tm["prompt_tokens"]),
  "{:.0f}", "decode / prefill, lowest over runs")
v("lap_ratio_hi", max(tm["decode_s_per_token_runs"]) / (min(tm["prefill_s_runs"]) / tm["prompt_tokens"]),
  "{:.0f}", "decode / prefill, highest over runs")
v("lap_prefill_runs", len(tm["prefill_s_runs"]), "{:.0f}", "prefill runs")
v("lap_decode_runs", len(tm["decode_s_per_token_runs"]), "{:.0f}", "decode runs")

print("== Code-completion trace (Azure 2023, token-mix scenario)")
tr = TRACE.result
v("az_requests", tr.requests, "{:,.0f}", "requests")
v("az_med_prompt", tr.median_prompt, "{:,.0f}", "median prompt tokens")
v("az_med_output", tr.median_output, "{:,.0f}", "median output tokens")
v("az_input_pct", 100 * tr.input_share, "{:.1f}", "input share of all tokens, %")
v("az_time_pct", 100 * tr.output_time_share, "{:.0f}", "output share of floor time, %")

print("== Napkin Math 2.2 (coding-agent resend scenarios)")
r30, r60 = R30.result, R60.result
v("rs_s0", R30.first_prompt, "{:,.0f}", "first prompt, tokens")
v("rs_delta", R30.appended, "{:,.0f}", "appended per turn, tokens")
v("rs_k", R30.output_per_turn, "{:.0f}", "output per call, tokens")
v("rs_turns", R30.turns, "{:.0f}", "turns")
v("rs_turns2", R60.turns, "{:.0f}", "doubled turns")
v("rs_resent", r30.resent_tokens, "{:,.0f}", "re-sent input, 30 turns")
v("rs_resent2", r60.resent_tokens, "{:,.0f}", "re-sent input, 60 turns")
v("rs_growth", r60.resent_tokens / r30.resent_tokens, "{:.1f}", "re-sent growth, 30 to 60 turns")
v("rs_app_growth", r60.appended_term / r30.appended_term, "{:.2f}", "appended-term growth")
v("rs_first_growth", r60.first_prompt_term / r30.first_prompt_term, "{:.1f}", "first-prompt-term growth")
v("rs_fresh", r30.reused_prefix_tokens, "{:,.0f}", "fresh prefill with reuse, 30 turns")
v("rs_fresh2", r60.reused_prefix_tokens, "{:,.0f}", "fresh prefill with reuse, 60 turns")
v("rs_fresh_growth", r60.reused_prefix_tokens / r30.reused_prefix_tokens, "{:.1f}", "fresh growth")
v("rs_reduction", r30.reuse_reduction, "{:.1f}", "reuse reduction, 30 turns")
v("rs_pre_s", m(r30.prefill_time, "s"), "{:.2f}", "prefill without reuse, s")
v("rs_pre_reuse_s", m(r30.reused_prefill_time, "s"), "{:.2f}", "prefill with reuse, s")
v("rs_dec_s", m(r30.decode_time, "s"), "{:.2f}", "decode, s")
v("rs_resent_k", r30.resent_tokens / 10 ** 3, "{:,.1f}", "figure: 30-turn resend, thousands")
v("rs_resent2_k", r60.resent_tokens / 10 ** 3, "{:,.0f}", "figure: 60-turn resend, thousands")
v("rs_fresh2_k", r60.reused_prefix_tokens / 10 ** 3, "{:,.1f}", "figure: 60-turn fresh prefill, thousands")

print("== Napkin Math 2.3 (whole-file and localized-edit calls)")
w, e = FULL.result, DIFF.result
v("ed_deadline_s", m(FULL.deadline, "s"), "{:.0f}", "call deadline, s")
v("ed_full_dm", FULL.instruction_tokens, "{:.0f}", "whole-file instruction tokens")
v("ed_full_k", FULL.output_tokens, "{:,.0f}", "whole-file output tokens")
v("ed_diff_dm", DIFF.instruction_tokens, "{:.0f}", "localized-edit instruction tokens")
v("ed_diff_k", DIFF.output_tokens, "{:.0f}", "localized-edit output tokens")
v("ed_full_s", m(w.total_time, "s"), "{:.2f}", "whole file, s")
v("ed_full_s1", m(w.total_time, "s"), "{:.1f}", "the same, one decimal (figure)")
v("ed_diff_s", m(e.total_time, "s"), "{:.2f}", "local edit, s")
v("ed_diff_s1", m(e.total_time, "s"), "{:.1f}", "the same, one decimal (figure)")
v("ed_speedup", (w.total_time / e.total_time).to("").magnitude, "{:.1f}", "ratio")
v("ed_full_dec_pct", 100 * w.output_time_share, "{:.1f}", "decode share of whole-file time, %")
v("ed_over_s", m(w.total_time - FULL.deadline, "s"), "{:.1f}", "whole file over the deadline, s")
v("ed_margin", e.deadline_margin, "{:.1f}", "local-edit margin under the deadline")
v("ed_apply", DIFF.stage_success["applies"], "{:.2f}", "edits that apply")
v("ed_pass", DIFF.stage_success["passes tests"], "{:.2f}", "pass rate given applied")
v("ed_apply_pct", 100 * DIFF.stage_success["applies"], "{:.0f}", "edits that apply, %")
v("ed_pass_pct", 100 * DIFF.stage_success["passes tests"], "{:.0f}", "pass rate given applied, %")
v("ed_yield_pct", 100 * e.acceptance, "{:.0f}", "trials ending accepted, %")

if __name__ == "__main__":
    write(__file__)
