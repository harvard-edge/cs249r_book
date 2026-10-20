#!/usr/bin/env python3
"""Every number on the Lecture 3 slides and notes (The Cognitive Brain), read from MLSysIM.

MLSysIM holds the inputs (the Embodied.* machine and its site record,
Literature.VlaSystems and Literature.EdgeInferenceOrin) and the calculations
(Scenarios.*, backed by the solvers in engine/solvers/proposer.py); the chapter
reads the same scenarios. This file only selects values, converts units, and
formats them; it types no numbers. Each value is recorded with v() under the
key the deck prints as \\val{key}; running this file writes images/numbers.tex.
make_figs.py imports it for the scenario objects and the plotted magnitudes.

Run from this folder: python3 calcs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Embodied, Scenarios, Literature = mlsysim.Embodied, mlsysim.Scenarios, mlsysim.Literature
ureg = mlsysim.core.units.ureg

RM = Embodied.MobileManipulator.WarehouseMobileManipulator
AISLE = Embodied.Scenario.WarehouseAisle
VLA = Literature.VlaSystems

FP16 = Scenarios.IntentModelEvaluationFp16
INT4 = Scenarios.IntentModelEvaluationInt4
TOKENS = Scenarios.ThreeViewVlaTokens
KV = Scenarios.ThreeViewKvHistory
MLPERF = Scenarios.OrinDevKitSingleStream
DECODE = Scenarios.IntentModelDecodeRoofline
QUANT = Scenarios.OpenVlaOrinQuantization
HALF = Scenarios.IntentModelHalfTokensFiveHertz
FIVE = Scenarios.IntentModelFiveHertz
TRAFFIC = Scenarios.WarehouseProposerTraffic
FAMILIES = Scenarios.IntentBackboneDecoderFamilies
EVIDENCE = Scenarios.ConveyorMugEvidenceHorizon
SLIP4 = Scenarios.ConveyorMugEvidenceHorizonFourfoldSlip
HALF_RATE = Scenarios.ConveyorMugEvidenceHorizonHalfRenewal
ENSEMBLE = Scenarios.ActTemporalEnsemble
REGIMES = Scenarios.ProposerConfidenceRegimes


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


f16, i4 = FP16.result, INT4.result
tr = TRAFFIC.result
fam = FAMILIES.result
ev, ev4, evh = EVIDENCE.result, SLIP4.result, HALF_RATE.result
sched = RM.chunk_schedule

print("== Hook: the Jetson AI Lab log line; the base's drive limit")
v("log_ms", float(VLA.OpenVlaOrinInt4LoggedFrameMs), "{:.1f}", "logged INT4 frame, ms")
v("meas_int4", m(RM.intent_reference_latency, "ms"), "{:.0f}", "measured INT4 mean, ms")
v("meas_int4_s", m(RM.intent_reference_latency, "s"), "{:.3f}", "the same, s")
v("v_drive", m(RM.base.max_velocity, "m/s"), "{:.1f}", "base drive limit, m/s")
v("hook_mm", m(RM.base.max_velocity * RM.intent_reference_latency, "mm"), "{:.0f}",
  "travel during one measured answer, mm")

print("== The running machine")
v("peak_fp16", m(RM.brain_soc.compute.precision_flops["fp16"], "TFLOPs/s"), "{:.0f}", "dense FP16 peak, TFLOP/s")
v("bw_peak", m(RM.brain_soc.memory.bandwidth, "GB/s"), "{:.0f}", "LPDDR5 peak, GB/s")
v("params_b", m(RM.intent_model.parameters, "param") / 10 ** 9, "{:.0f}", "intent-model parameters, billions")
v("chunk_params_m", m(RM.chunk_model.parameters, "param") / 10 ** 6, "{:.0f}", "chunk-policy parameters, millions")
v("intent_hz", m(RM.intent_rate, "Hz"), "{:.0f}", "intent rate, Hz")
v("next_hz", m(RM.intent_rate, "Hz") + 1, "{:.0f}", "next whole-hertz rate, Hz")
v("next_period", m(1 / (RM.intent_rate + 1 * ureg.Hz), "ms"), "{:.0f}", "its period, ms")
v("chunk_hz", m(RM.chunk_rate, "Hz"), "{:.0f}", "chunk renewal rate, Hz")
v("chunk_infer", m(RM.chunk_inference_latency, "ms"), "{:.0f}", "chunk inference target, ms")
v("horizon", RM.chunk_horizon, "{:.0f}", "setpoints per chunk H")
v("step_ms", m(RM.chunk_step, "ms"), "{:.0f}", "setpoint step, ms")
v("tick_khz", m(RM.permission_rate, "kHz"), "{:.0f}", "permission tick, kHz")
v("eta_pct", 100 * RM.dram_efficiency, "{:.0f}", "sustained fraction of peak bandwidth, %")
v("u_pct", 100 * RM.intent_compute_utilization, "{:.0f}", "achieved fraction of FP16 peak, %")
v("belt", m(AISLE.v_conveyor, "m/s"), "{:.2f}", "conveyor speed, m/s")
v("tol_mm", m(AISLE.grasp_tolerance, "mm"), "{:.0f}", "grasp tolerance, mm")
v("err0_mm", m(AISLE.grasp_initial_error, "mm"), "{:.0f}", "initial handle error, mm")
v("t_chunk", m(RM.chunk_period, "ms"), "{:.0f}", "chunk period, ms")
v("t_intent", m(1 / RM.intent_rate, "ms"), "{:.0f}", "intent period, ms")
v("d_action", RM.intent_model.action_dim, "{:.0f}", "action values per answer")

print("== Tokens (three-camera step, OpenVLA config.json)")
v("frame_px", TOKENS.frame_px, "{:.0f}", "frame size, px")
v("patch_px", TOKENS.patch_px, "{:.0f}", "patch size, px")
v("patches_side", TOKENS.patches_per_side, "{:.0f}", "patches per side")
v("patches_view", TOKENS.patches_per_view, "{:.0f}", "patch tokens per view")
v("views", TOKENS.views, "{:.0f}", "camera views")
v("views_words", TOKENS.views, "words", "the same, spelled")
v("visual_tokens", TOKENS.visual_tokens, "{:,.0f}", "visual tokens, three views")
v("prompt_tok", TOKENS.prompt_tokens, "{:.0f}", "prompt tokens")
v("state_tok", TOKENS.state_tokens, "{:.0f}", "joint-state tokens")
v("ctx", TOKENS.context_tokens, "{:.0f}", "rounded context S")
v("d_model", RM.intent_model.hidden_dim, "{:,.0f}", "hidden width")
v("layers", RM.intent_model.layers, "{:.0f}", "layers")
v("elem_bytes", m(TOKENS.element_bytes, "byte"), "{:.0f}", "bytes per FP16 value")
v("embed_mb", m(TOKENS.embedding_memory, "MB"), "{:.2f}", "embeddings for S = 800, MB")
v("w_fp16_gb", m(f16.weight_memory, "GB"), "{:.0f}", "FP16 weights, GB")
v("w_over_embed", round(ratio(f16.weight_memory, TOKENS.embedding_memory), -3), "{:,.0f}",
  "weights over one step of embeddings (nearest thousand)")
v("s_in", INT4.tokens_in, "{:.0f}", "intent-model input tokens")

print("== KV cache (hypothetical retained history)")
v("kv_tok_bytes", m(TOKENS.kv_per_token, "byte"), "{:,.0f}", "KV per token, bytes")
v("kv_tok_mb", m(TOKENS.kv_per_token, "MB"), "{:.2f}", "KV per token, MB")
v("kv_step_mb", m(KV.kv_per_step, "MB"), "{:.0f}", "KV per 800-token step, MB")
v("hist_s", m(KV.history_window, "s"), "{:.0f}", "history window, s")
v("hist_hz", m(KV.history_rate, "Hz"), "{:.0f}", "history rate, Hz")
v("kv_hist_gb", m(KV.history, "GB"), "{:.1f}", "KV history, GB")
v("combined_gb", m(KV.combined, "GB"), "{:.1f}", "history plus FP16 weights, GB")
v("soc_gb", m(KV.module_memory, "GB"), "{:.1f}", "module memory, GB")
v("soc_gib", m(KV.module_memory, "GiB"), "{:.0f}", "module memory, GiB")

print("== MLPerf Inference edge on the Orin (single stream, p90)")
lat = {b: m(MLPERF.latest(b), "ms") for b, _, _ in MLPERF.benchmarks}
v("n_bench", len(MLPERF.benchmarks), "{:.0f}", "reference models")
v("n_bench_words", len(MLPERF.benchmarks), "words", "the same, spelled")
v("n_fit", MLPERF.count_within(RM.chunk_period), "{:.0f}", "models inside the chunk period")
v("n_fit_words", MLPERF.count_within(RM.chunk_period), "words", "the same, spelled")
v("mlp_resnet", lat["resnet50"], "{:.2f}", "ResNet-50, ms")
v("mlp_bert", lat["bert-99"], "{:.1f}", "BERT, ms")
v("mlp_retina", lat["retinanet"], "{:.1f}", "RetinaNet, ms")
v("mlp_rnnt", lat["rnnt"], "{:.0f}", "RNN-T, ms")
v("gptj_s", m(MLPERF.latest("gptj-99"), "s"), "{:.1f}", "GPT-J, s")
v("gptj_early_s", m(MLPERF.earliest("gptj-99"), "s"), "{:.1f}", "GPT-J one round earlier, s")
v("gptj_over_intent", ratio(MLPERF.latest("gptj-99"), 1 / RM.intent_rate), "{:.1f}", "GPT-J over the intent period")
v("mlperf_span", round(MLPERF.span, -3), "{:,.0f}", "slowest over fastest (nearest thousand)")

print("== Memory wall and ceilings")
v("b_sus", m(tr.sustained_bandwidth, "GB/s"), "{:.1f}", "sustained bandwidth, GB/s")
v("sweep16", m(f16.t_weight_sweep, "ms"), "{:.1f}", "FP16 sweep, ms")
v("sweep16_0", m(f16.t_weight_sweep, "ms"), "{:.0f}", "the same, whole ms")
v("wall16", m(f16.weight_sweep_rate, "Hz"), "{:.1f}", "FP16 ceiling, Hz")
v("tick_over", tr.tick_over_sweep_rate, "{:.0f}", "permission tick over the FP16 ceiling")
v("bpp16", m(FP16.bytes_per_param, "byte/param"), "{:.0f}", "FP16 bytes per parameter")
v("bpp16_words", m(FP16.bytes_per_param, "byte/param"), "words", "the same, spelled")
v("bpp4", m(INT4.bytes_per_param, "byte/param"), "{:.1f}", "INT4 bytes per parameter")
v("int4_factor_words", ratio(FP16.bytes_per_param, INT4.bytes_per_param), "words", "INT4 ceiling over FP16")
v("chunk_ceiling", round(m(tr.chunk_sweep_rate, "Hz"), -2), "{:.0f}", "chunk-policy FP16 ceiling (nearest hundred), Hz")
v("wall4", m(tr.intent_sweep_rate_deployed, "Hz"), "{:.1f}", "INT4 ceiling, Hz")

print("== Roofline")
v("ridge", m(f16.ridge_intensity, "flop/byte"), "{:.1f}", "ridge, FLOP/byte")
v("decode_i", m(f16.decode_intensity, "flop/byte"), "{:.0f}", "decode intensity at FP16, FLOP/byte")
v("decode_tflops", m(DECODE.attainable, "TFLOPs/s"), "{:.3f}", "decode attainable, TFLOP/s")
v("decode_pct", 100 * DECODE.peak_fraction, "{:.2f}", "decode share of peak, %")
v("prefill_i", m(f16.prefill_intensity, "flop/byte"), "{:.0f}", "prefill intensity, FLOP/byte")

print("== Worked example 1: one evaluation")
pf = m(f16.prefill_flops, "flop")
exp10 = len(f"{pf:.0f}") - 1
v("prefill_mant", pf / 10 ** exp10, "{:.2f}", "prefill FLOPs, mantissa")
v("prefill_exp", exp10, "{:.0f}", "prefill FLOPs, power of ten")
v("achieved", m(f16.achieved_rate, "TFLOPs/s"), "{:.1f}", "achieved prefill rate, TFLOP/s")
v("prefill_1", m(i4.t_prefill, "ms"), "{:.1f}", "prefill, ms")
v("prefill", m(i4.t_prefill, "ms"), "{:.0f}", "prefill, whole ms")
v("decode_steps", i4.decode_steps, "{:.0f}", "decode sweeps")
v("decode16", m(f16.t_decode, "ms"), "{:.0f}", "FP16 decode, ms")
v("t_fp16", m(f16.t_total, "ms"), "{:.0f}", "FP16 total, ms")
v("f_fp16", m(f16.f_max, "Hz"), "{:.1f}", "FP16 rate, Hz")
v("w_int4_gb", m(i4.weight_memory, "GB"), "{:.1f}", "INT4 weights, GB")
v("sweep4", m(i4.t_weight_sweep, "ms"), "{:.1f}", "INT4 sweep, ms")
v("decode4", m(i4.t_decode, "ms"), "{:.0f}", "INT4 decode, ms")
v("t_int4", m(i4.t_total, "ms"), "{:.0f}", "INT4 total, ms")

print("== Quantization, computed against measured")
v("q_c16", m(QUANT.computed("FP16"), "ms"), "{:.0f}", "computed FP16, ms")
v("q_c8", m(QUANT.computed("8-bit"), "ms"), "{:.0f}", "computed 8-bit, ms")
v("q_c4", m(QUANT.computed("INT4"), "ms"), "{:.0f}", "computed INT4, ms")
v("q_m16", m(QUANT.measured["FP16"], "ms"), "{:.0f}", "measured FP16, ms")
v("q_m8", m(QUANT.measured["8-bit"], "ms"), "{:.0f}", "measured 8-bit, ms")
v("q_m4", m(QUANT.measured["INT4"], "ms"), "{:.0f}", "measured INT4, ms")
v("q_s16", 100 * QUANT.scores["FP16"], "{:.1f}", "action score FP16, %")
v("q_s8", 100 * QUANT.scores["8-bit"], "{:.1f}", "action score 8-bit, %")
v("q_s4", 100 * QUANT.scores["INT4"], "{:.1f}", "action score INT4, %")
v("q_gap16", 100 * abs(QUANT.gap("FP16")), "{:.1f}", "FP16 underestimate, %")
v("q_gap8", 100 * abs(QUANT.gap("8-bit")), "{:.1f}", "8-bit gap, %")
v("q_gap4", 100 * abs(QUANT.gap("INT4")), "{:.1f}", "INT4 gap, %")
v("q_speedup", QUANT.measured_speedup, "{:.1f}", "measured INT4 speedup over FP16")
v("q_drop", 100 * QUANT.score_drop, "{:.1f}", "action-score drop, points")
v("ovla_bf16", 100 * float(VLA.OpenVlaSuccessBf16), "{:.1f}", "OpenVLA success bfloat16, %")
v("ovla_int8", 100 * float(VLA.OpenVlaSuccessInt8), "{:.1f}", "OpenVLA success int8, %")
v("ovla_int4", 100 * float(VLA.OpenVlaSuccessInt4), "{:.1f}", "OpenVLA success int4, %")
v("ovla_int8_hz", float(VLA.OpenVlaInt8RateHz), "{:.1f}", "int8 control rate, Hz")
v("ovla_train_hz", float(VLA.OpenVlaTrainingRateHz), "{:.0f}", "training-data control rate, Hz")
v("prefill_ceiling", m(i4.prefill_ceiling, "Hz"), "{:.1f}", "weight-only ceiling 1 / prefill, Hz")

print("== Estimate: half the input tokens against 5 Hz")
v("p5_hz", m(FIVE.target_rate, "Hz"), "{:.0f}", "target rate, Hz")
v("p5_ms", m(FIVE.period, "ms"), "{:.0f}", "its period, ms")
v("half_prefill", m(HALF.evaluation.result.t_prefill, "ms"), "{:.0f}", "prefill at half the tokens, ms")
v("half_total", m(HALF.evaluation.result.t_total, "ms"), "{:.0f}", "total at half the tokens, ms")
v("half_over", m(-HALF.slack, "ms"), "{:.0f}", "over the period by, ms")
v("fit_prefill", m(FIVE.prefill_budget, "ms"), "{:.0f}", "prefill that would fit, ms")
v("fit_pct", 100 * FIVE.token_fraction, "{:.0f}", "token share that would fit, %")
v("fit_tokens", FIVE.tokens_within, "{:.0f}", "tokens that would fit")

print("== Chunk supply")
v("setpoint_hz", m(tr.setpoint_rate, "Hz"), "{:.0f}", "setpoints per second")
v("demand", m(tr.single_step_demand, "GB/s"), "{:.0f}", "one sweep per setpoint, GB/s")
v("overload", tr.single_step_overload, "{:.1f}", "that demand over sustained")
v("chunked", m(tr.chunked_demand, "GB/s"), "{:.0f}", "one sweep per chunk, GB/s")
v("chunked_pct", 100 * tr.chunked_share, "{:.1f}", "chunked share of sustained, %")
v("react", m(ev.age_at_replacement, "ms"), "{:.0f}", "age at replacement, ms")

print("== Decoder families on the INT4 backbone")
v("binned_tok", fam.binned_tokens, "{:.0f}", "binned tokens")
v("fast_tok", FAMILIES.fast_tokens, "{:.0f}", "FAST tokens")
v("expert_mb", m(fam.expert_weights, "MB"), "{:.0f}", "expert weights, MB")
v("expert_steps", FAMILIES.expert_steps, "{:.0f}", "expert steps")
v("t_binned_s", m(fam.t_binned, "s"), "{:.1f}", "binned, s")
v("t_fast", m(fam.t_fast, "ms"), "{:.0f}", "FAST, ms")
v("t_par", m(fam.t_parallel, "ms"), "{:.0f}", "parallel head, ms")
v("t_expert", m(fam.t_expert, "ms"), "{:.0f}", "flow expert, ms")
v("dec_spread", fam.binned_over_expert, "{:.0f}", "binned over expert")
v("prefill_over_period", fam.prefill_over_period, "{:.1f}", "prefill over the renewal period")

print("== Worked example 2: renewal gap and bus share")
v("amort", m(tr.amortized_sweep, "ms"), "{:.1f}", "FP16 sweep per setpoint, ms")
v("renewal_gap", tr.renewal_gap, "{:.1f}", "renewal periods per intent evaluation")
v("chunk_w_gb", m(tr.chunk_weights, "GB"), "{:.2f}", "chunk-policy weights, GB")
v("chunk_sweep", m(tr.chunk_sweep, "ms"), "{:.2f}", "chunk-policy sweep, ms")
v("chunk_sweep_pct", 100 * tr.chunk_sweep_share, "{:.1f}", "chunk sweep share of its period, %")
v("intent_bw", m(tr.intent_traffic, "GB/s"), "{:.0f}", "intent traffic, GB/s")
v("chunk_bw", m(tr.chunk_traffic, "GB/s"), "{:.1f}", "chunk traffic, GB/s")
v("prop_bw", m(tr.proposer_traffic, "GB/s"), "{:.1f}", "proposer traffic, GB/s")
v("prop_pct", 100 * tr.proposer_share, "{:.0f}", "proposer share of sustained, %")

print("== Published two-speed designs")
helix_lo, helix_hi, helix_fast = (float(VLA.HelixSlowRateLowHz), float(VLA.HelixSlowRateHighHz),
                                  float(VLA.HelixFastRateHz))
gr_slow, gr_fast = float(VLA.Gr00tN1SlowRateHz), float(VLA.Gr00tN1FastRateHz)
v("helix_slow_b", float(VLA.HelixSlowParams) / 10 ** 9, "{:.0f}", "Helix slow model, billions")
v("helix_fast_m", float(VLA.HelixFastParams) / 10 ** 6, "{:.0f}", "Helix fast policy, millions")
v("helix_dof", float(VLA.HelixDegreesOfFreedom), "{:.0f}", "Helix degrees of freedom")
v("helix_lo", helix_lo, "{:.0f}", "Helix slow rate, low, Hz")
v("helix_hi", helix_hi, "{:.0f}", "Helix slow rate, high, Hz")
v("helix_fast", helix_fast, "{:.0f}", "Helix fast rate, Hz")
v("gr_slow", gr_slow, "{:.0f}", "GR00T N1 slow rate, Hz")
v("gr_fast", gr_fast, "{:.0f}", "GR00T N1 fast rate, Hz")
v("helix_ratio_lo", helix_fast / helix_hi, "{:.0f}", "Helix fast over slow, low")
v("helix_ratio_hi", helix_fast / helix_lo, "{:.0f}", "Helix fast over slow, high")
v("gr_ratio", gr_fast / gr_slow, "{:.0f}", "GR00T N1 fast over slow")
v("our_ratio", ratio(RM.chunk_rate, RM.intent_rate), "{:.0f}", "this machine, fast over slow")
DUAL = [("Helix", helix_lo, helix_hi, helix_fast), ("GR00T N1", gr_slow, gr_slow, gr_fast),
        ("This machine", m(RM.intent_rate, "Hz"), m(RM.intent_rate, "Hz"), m(RM.chunk_rate, "Hz"))]

print("== Evidence horizon and the timing strip (Napkin Math 3.3)")
v("slip", m(AISLE.a_conveyor_slip, "m/s**2"), "{:.2f}", "residual slip, m/s^2")
v("slip_1", m(AISLE.a_conveyor_slip, "m/s**2"), "{:.1f}", "the same, one decimal")
v("margin_mm", m(ev.margin, "mm"), "{:.0f}", "margin, mm")
v("untracked_ms", m(ev.untracked_deadline, "ms"), "{:.0f}", "untracked deadline, ms")
v("tau_ev", m(ev.horizon, "ms"), "{:.0f}", "evidence horizon, ms")
v("drift_react", m(ev.drift_at_replacement, "mm"), "{:.1f}", "drift at replacement, mm")
v("span_ms", m(ev.chunk_span, "ms"), "{:.0f}", "chunk span, ms")
v("drift_span", m(ev.drift_at_chunk_end, "mm"), "{:.0f}", "drift at the chunk's end, mm")
v("err_end", m(ev.error_at_chunk_end, "mm"), "{:.0f}", "error at the chunk's end, mm")
v("outlast", m(ev.outlast, "ms"), "{:.0f}", "chunk outlasts its evidence by, ms")
v("h_min", ev.min_supply_setpoints, "{:.0f}", "setpoints that bridge a renewal")
v("h_min_words", ev.min_supply_setpoints, "words", "the same, in words")
v("per_renewal", sched["setpoints_per_renewal"], "{:.0f}", "setpoints per renewal")
v("overlap", sched["overlapping_chunks"], "{:.0f}", "chunks covering one step")
v("slip_factor_words", ratio(SLIP4.a_slip, EVIDENCE.a_slip), "words", "slip multiple, spelled")
v("slip4", m(SLIP4.a_slip, "m/s**2"), "{:.1f}", "quadrupled slip, m/s^2")
v("tau_ev4", m(ev4.horizon, "ms"), "{:.0f}", "horizon at four times the slip, ms")
v("spare4", m(ev4.horizon_spare, "ms"), "{:.0f}", "spare after the age at replacement, ms")
v("drift_react4", m(ev4.drift_at_replacement, "mm"), "{:.1f}", "drift at replacement with four times the slip, mm")
v("half_hz", m(HALF_RATE.renewal_rate, "Hz"), "{:.0f}", "half renewal rate, Hz")
v("react_half", m(evh.age_at_replacement, "ms"), "{:.0f}", "age at replacement at half rate, ms")
v("h_min_half", evh.min_supply_setpoints, "{:.0f}", "setpoints that bridge it")

print("== ACT temporal ensembling")
v("act_chunk", ENSEMBLE.predictions, "{:.0f}", "overlapping predictions (ACT chunk size)")
v("ens_ratio", ENSEMBLE.result["oldest_over_newest"], "{:.1f}", "oldest weight over newest")

print("== Confidence against success (illustrative)")
rows = {name: (c, a) for name, c, a in REGIMES.regimes}
for key, name in (("in", "In distribution"), ("near", "Near OOD"), ("sensor", "Sensor degraded"), ("far", "Far OOD")):
    v(f"conf_{key}", rows[name][0], "{:.2f}", f"{name}: confidence")
    v(f"succ_{key}", rows[name][1], "{:.2f}", f"{name}: task success")
v("gap_in", REGIMES.gap("In distribution"), "{:.2f}", "in-distribution gap")
v("near_fail_pct", 100 * (1 - rows["Near OOD"][1]), "{:.0f}", "near-OOD attempts that fail, %")

print("== Synthesis")
v("decode4_syn", m(i4.t_total - i4.t_prefill, "ms"), "{:.0f}", "decode share of the INT4 evaluation, ms")

if __name__ == "__main__":
    write(__file__)
