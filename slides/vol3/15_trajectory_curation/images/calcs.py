#!/usr/bin/env python3
"""Every number on the Volume III Lecture 15 slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations, and the chapter reads the same
entries: the month of failures triaged by counterfactual probes and the price
of fine-tuning on all of them (Scenarios.RepairAgentFailureTriage), random
splits of runs (EightAttemptRunSplit and, for the estimate,
FiveAttemptRunSplitOfTwoThousand), the MinHash screen
(TrajectoryMinHashScreen), the admission route in two orders
(DemonstrationAdmissionOrder and its SlowAudit variant), the three
denominators of a retained pool (RecoveryPoolDenominators), attempt-group
contrast (EightAttemptGroupContrast), the staging burst
(VerifierTimeoutStagingBurst), and the cohort's dispositions
(CurationCohortDispositions), all backed by engine/solvers/curation.py.
The one other source is the deck's own data (data/): counts measured on the
public nebius/SWE-agent-trajectories corpus (corpus_stats.json, written by
corpus_stats.py). Quoted excerpts (dataset card, blog post, tool output)
stay verbatim in the .tex. This file only selects values, converts units, and
formats them; it types no numbers. Each value is recorded with v() under the
key the deck prints as \\val{key}; running this file writes images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.physics import calc_split_straddle  # noqa: E402

Scenarios = mlsysim.Scenarios
Literature = mlsysim.Literature
HERE = Path(__file__).resolve().parent
CORPUS = json.loads((HERE / "data" / "corpus_stats.json").read_text())

TRI_S = Scenarios.RepairAgentFailureTriage
SP_S = Scenarios.EightAttemptRunSplit
EX_S = Scenarios.FiveAttemptRunSplitOfTwoThousand
MH_S = Scenarios.TrajectoryMinHashScreen
AD_S = Scenarios.DemonstrationAdmissionOrder
AV_S = Scenarios.DemonstrationAdmissionOrderSlowAudit
DN_S = Scenarios.RecoveryPoolDenominators
GC_S = Scenarios.EightAttemptGroupContrast
ST_S = Scenarios.VerifierTimeoutStagingBurst
CO_S = Scenarios.CurationCohortDispositions

TRI = TRI_S.result
SP = SP_S.result
EX = EX_S.result
AD = AD_S.result
AV = AV_S.result
DN = DN_S.result
ST = ST_S.result


def pct(x):
    return x * 100


def sec(q):
    return q.to("second").magnitude


def exponent(x):
    return math.floor(math.log10(x))


def mantissa(x):
    return x / 10 ** exponent(x)


# Figure helpers: the same calculations, for curves the figures sweep.
def straddle(attempts):
    """Straddle share of a random run split at the lecture's training fraction."""
    return calc_split_straddle(SP_S.train_fraction, attempts)


def lsh_p(similarity):
    return MH_S.candidate_probability(similarity)


def yield_any(p):
    return GC_S.success_yield(p)


def mixed(p):
    return GC_S.mixed_fraction(p)


print("== Triage and the fine-tuning invoice (Scenarios.RepairAgentFailureTriage)")
_flops = TRI.training_flops.to("flop").magnitude
_rate = TRI.sustained_rate.to("flop/s").magnitude
v("tri_runs", TRI_S.runs, "{:,.0f}", "trajectories a month")
v("tri_fail_pct", pct(TRI_S.failure_rate), "{:.0f}", "failing share, %")
v("tri_fail", TRI.failures, "{:,.0f}", "failures a month")
v("tri_iface", TRI.removed["interface"], "{:,.0f}", "schema probe removes")
v("tri_iface_pct", pct(TRI_S.probe_shares["interface"]), "{:.0f}", "the same, %")
v("tri_rt", TRI.removed["runtime"], "{:,.0f}", "runtime probe removes")
v("tri_rt_pct", pct(TRI_S.probe_shares["runtime"]), "{:.0f}", "the same, %")
v("tri_system", TRI.system_failures, "{:,.0f}", "repaired below the weights")
v("tri_residual", TRI.residual, "{:,.0f}", "survive every probe")
v("tri_seq", TRI_S.sequence_tokens, "{:,.0f}", "tokens per sequence")
v("tri_epochs", TRI_S.epochs, "{:.0f}", "epochs")
v("tri_params_b", TRI_S.workload.parameters.to("Gparam").magnitude, "{:.1f}", "parameters, billions")
v("tri_fpt", float(Literature.Chinchilla.ComputeConstant), "{:.0f}", "training FLOP per parameter-token")
v("tri_dev", TRI_S.devices, "{:.0f}", "accelerators")
v("tri_peak", TRI_S.hardware.compute.peak_flops.to("TFLOPs/s").magnitude, "{:.0f}", "dense 16-bit peak, TFLOP/s")
v("tri_mfu", TRI_S.utilization, "{:.2f}", "model FLOPs utilization")
v("tri_mfu_pct", pct(TRI_S.utilization), "{:.0f}", "the same, %")
v("tri_price", TRI_S.hourly_price.to("USD/hour").magnitude, "{:.0f}", "USD per accelerator-hour")
v("tri_tokens", TRI.training_tokens.to("token").magnitude, "{:,.0f}", "training tokens")
v("tri_tokens_m", TRI.training_tokens.to("token").magnitude / 10 ** 6, "{:.1f}", "the same, millions")
v("tri_flops_mant", mantissa(_flops), "{:.2f}", "training FLOPs, mantissa")
v("tri_flops_exp", exponent(_flops), "{:.0f}", "training FLOPs, power of ten")
v("tri_rate_mant", mantissa(_rate), "{:.2f}", "sustained FLOP/s, mantissa")
v("tri_rate_exp", exponent(_rate), "{:.0f}", "sustained FLOP/s, power of ten")
v("tri_wall_h", TRI.wall_time.to("hour").magnitude, "{:.2f}", "wall-clock hours")
v("tri_wall_nomfu", TRI.wall_time.to("hour").magnitude * TRI_S.utilization, "{:.2f}", "hours at full peak (the slip)")
v("tri_dev_h", TRI.device_hours.to("hour").magnitude, "{:.1f}", "accelerator-hours")
v("tri_cost", TRI.cost.to("USD").magnitude, "{:.0f}", "invoice, USD")
v("tri_cost_exact", TRI.cost.to("USD").magnitude, "{:.2f}", "invoice, USD, to the cent")
v("tri_res_cost", TRI.residual_cost.to("USD").magnitude, "{:.0f}", "residual-only invoice, USD")

print("== Measured corpus (data/corpus_stats.json)")
_rc = CORPUS["recovery"]
_dup = CORPUS["dup"]
_mix = CORPUS["outcome_mix"]
_hist = {int(k): n for k, n in CORPUS["runs_per_task_hist"].items()}
_resolved_runs = _rc["direct"] + _rc["recovery"]
v("c_runs", CORPUS["runs"], "{:,.0f}", "runs")
v("c_tasks", CORPUS["tasks"], "{:,.0f}", "issues")
v("c_repos", CORPUS["repos"], "{:,.0f}", "repositories")
v("c_resolved", CORPUS["resolved"], "{:,.0f}", "resolved runs")
v("c_resolved_pct", pct(CORPUS["resolved"] / CORPUS["runs"]), "{:.1f}", "the same, %")
v("c_median", CORPUS["runs_per_task"]["median"], "{:.0f}", "median runs per issue")
v("c_max", CORPUS["runs_per_task"]["max"], "{:.0f}", "most runs on one issue")
v("c_cap_bar", max(_hist), "{:.0f}", "histogram's last bar: this many runs or more")
v("c_at_cap", _hist[max(_hist)], "{:,.0f}", "issues in that bar")
v("c_straddle_pct", pct(CORPUS["straddle"]["measured"]), "{:.1f}", "issues on both sides of a run split, %")
v("c_straddle_tasks", CORPUS["straddle"]["tasks"], "{:,.0f}", "the same, issues")
v("c_pred_pct", pct(CORPUS["straddle"]["predicted"]), "{:.1f}", "formula per issue's run count, %")
v("c_test_tasks", CORPUS["group_split"]["test_tasks"], "{:,.0f}", "test issues of the group split")
v("c_test_repo_pct", pct(CORPUS["group_split"]["test_repo_shared"] / CORPUS["group_split"]["test_tasks"]),
  "{:.1f}", "test issues whose repository is in training, %")
v("c_sig", _dup["resolved_with_signature"], "{:,.0f}", "resolved patches with a signature")
v("c_dup", _dup["same_task_dup"], "{:,.0f}", "with a same-issue near-duplicate")
v("c_dup_pct", pct(_dup["same_task_dup"] / _dup["resolved_with_signature"]), "{:.1f}", "the same, %")
v("c_xpairs", _dup["cross_task_pairs"], "{:.0f}", "near-duplicate pairs across issues")
v("c_xrepo", _dup["cross_repo_pairs"], "{:.0f}", "of them across repositories")
v("c_ai_masked", CORPUS["mask_by_role"]["ai|True"], "{:,.0f}", "model messages marked mask=True")
v("c_ai_total", CORPUS["mask_by_role"]["ai|True"] + CORPUS["mask_by_role"].get("ai|False", 0), "{:,.0f}",
  "model messages")
v("c_refusal_pct", pct(_rc["runs_with_refusal"] / CORPUS["runs"]), "{:.1f}", "runs with a refused edit, %")
v("c_rec", _rc["recovery"], "{:,.0f}", "resolved runs with a refused edit")
v("c_rec_pct", pct(_rc["recovery"] / _resolved_runs), "{:.1f}", "the same, % of resolved")
v("c_rec_chars_pct", pct(_rc["recovery_chars"] / (_rc["recovery_chars"] + _rc["direct_chars"])), "{:.1f}",
  "their share of resolved characters, %")
v("c_rec_ai_pct", pct(_rc["recovery_ai"] / (_rc["recovery_ai"] + _rc["direct_ai"])), "{:.1f}",
  "their share of model-written characters, %")
_d_mean = _rc["direct_chars"] / _rc["direct"]
_r_mean = _rc["recovery_chars"] / _rc["recovery"]
v("c_direct_mean", _d_mean, "{:,.0f}", "mean characters, direct runs")
v("c_rec_mean", _r_mean, "{:,.0f}", "mean characters, recovery runs")
v("c_len_ratio", _r_mean / _d_mean, "{:.2f}", "recovery runs are longer by, x")
v("c_cap_rec_pct", pct(_rc["cap30_recovery"] / _rc["cap30_kept"]), "{:.1f}", "recovery share after the shortest-run cap, %")
v("c_allfail_pct", pct(_mix["all_fail"] / CORPUS["tasks"]), "{:.1f}", "issues that never succeed, %")
v("c_mixed_pct", pct(_mix["mixed"] / CORPUS["tasks"]), "{:.1f}", "issues with mixed outcomes, %")
v("c_allpass_pct", pct(_mix["all_pass"] / CORPUS["tasks"]), "{:.1f}", "issues that always succeed, %")

print("== Random split of runs (Scenarios.EightAttemptRunSplit)")
v("sp_k", SP_S.attempts, "{:.0f}", "attempts per task")
v("sp_f", SP_S.train_fraction, "{:.1f}", "training probability per run")
v("sp_q", 1 - SP_S.train_fraction, "{:.1f}", "test probability per run")
v("sp_f_pct", pct(SP_S.train_fraction), "{:.0f}", "training share, %")
v("sp_q_pct", pct(1 - SP_S.train_fraction), "{:.0f}", "test share, %")
v("sp_whole", SP.whole, "{:.3f}", "all attempts on one side")
v("sp_straddle", pct(SP.straddle), "{:.1f}", "tasks straddling, %")
v("sp_straddle_r", pct(SP.straddle), "{:.0f}", "the same, whole percent")
v("sp_two_pct", pct(SP.curve[2]), "{:.0f}", "straddling with two attempts, %")

print("== Estimate: 5 attempts, 90/10 (Scenarios.FiveAttemptRunSplitOfTwoThousand)")
v("ex_tasks", EX_S.tasks, "{:,.0f}", "repair tasks")
v("ex_k", EX_S.attempts, "{:.0f}", "attempts per task")
v("ex_f", EX_S.train_fraction, "{:.1f}", "training probability per run")
v("ex_q", 1 - EX_S.train_fraction, "{:.1f}", "test probability per run")
v("ex_f_pct", pct(EX_S.train_fraction), "{:.0f}", "training share, %")
v("ex_q_pct", pct(1 - EX_S.train_fraction), "{:.0f}", "test share, %")
v("ex_q_exp", round(math.log10((1 - EX_S.train_fraction) ** EX_S.attempts)), "{:.0f}",
  "all runs to test, power of ten")
v("ex_whole", EX.whole, "{:.4f}", "all attempts on one side")
v("ex_straddle", pct(EX.straddle), "{:.2f}", "tasks straddling, %")
v("ex_tasks_straddle", EX.straddling_tasks, "{:.0f}", "tasks straddling")
v("ex_sib_k", EX_S.attempts - 1, "{:.0f}", "siblings of one test run")
v("ex_sib", EX.sibling_in_training, "{:.4f}", "a test run has a sibling in training")
v("ex_sib_pct", pct(EX.sibling_in_training), "{:.2f}", "the same, %")
v("ex_major", EX.majority_attempts, "{:.0f}", "attempts at which most tasks straddle")
v("ex_major_pct", pct(EX.majority_straddle), "{:.1f}", "straddle share there, %")

print("== MinHash screen (Scenarios.TrajectoryMinHashScreen)")
v("mh_k", MH_S.hashes, "{:.0f}", "signature components")
v("mh_bytes", MH_S.signature_bytes.to("byte").magnitude, "{:.0f}", "bytes per signature")
v("mh_records", MH_S.records, "{:,.0f}", "records")
v("mh_mb", MH_S.index_bytes.to("MB").magnitude, "{:.1f}", "signature matrix, MB")
v("mh_se", MH_S.max_standard_error, "{:.3f}", "largest standard error")
v("mh_bands", MH_S.bands, "{:.0f}", "bands")
v("mh_rows", MH_S.rows, "{:.0f}", "rows per band")
_j_lo, _j_hi = MH_S.probe_similarities
v("mh_j_lo", _j_lo, "{:.1f}", "loose pair's Jaccard similarity")
v("mh_j_hi", _j_hi, "{:.1f}", "near-duplicate's Jaccard similarity")
v("mh_p_lo", pct(MH_S.candidate_probability(_j_lo)), "{:.0f}", "loose pairs sent to review, %")
v("mh_p_hi", pct(MH_S.candidate_probability(_j_hi)), "{:.0f}", "near-duplicates sent to review, %")

print("== Admission order (Scenarios.DemonstrationAdmissionOrder)")
for _name, (_p, _t) in AD_S.checks.items():
    v(f"ad_pass_{_name}", pct(_p), "{:.0f}", f"{_name} check passes, %")
    v(f"ad_time_{_name}", sec(_t), "{:.2f}" if sec(_t) < 1 else "{:.0f}", f"{_name} check slot time, s")
v("ad_cands", AD_S.candidates, "{:,.0f}", "candidate records")
v("ad_rej_pct_process", pct(1 - AD_S.checks["process"][0]), "{:.0f}", "process audit rejects, %")
v("ad_rej_pct_outcome", pct(1 - AD_S.checks["outcome"][0]), "{:.0f}", "tests reject, %")
v("ad_rej_process", sec(AD.cost_per_rejection["process"]), "{:.0f}", "process audit, s per rejection")
v("ad_rej_outcome", sec(AD.cost_per_rejection["outcome"]), "{:.1f}", "tests, s per rejection")
_best, _worst = AD.best_order, 1 - AD.best_order
v("ad_t_best", sec(AD.time_per_attempted[_best]), "{:.2f}", "process first, s per attempted")
v("ad_t_worst", sec(AD.time_per_attempted[_worst]), "{:.2f}", "tests first, s per attempted")
v("ad_ta_best", sec(AD.time_per_admitted[_best]), "{:.2f}", "process first, s per admitted")
v("ad_ta_worst", sec(AD.time_per_admitted[_worst]), "{:.2f}", "tests first, s per admitted")
v("ad_yield", pct(AD.route_yield), "{:.2f}", "route yield, %")
v("ad_admitted", AD.admitted, "{:,.0f}", "records admitted")
v("ad_saving", pct(AD.saving), "{:.1f}", "verifier time saved, %")
v("av_time_process", sec(AV_S.checks["process"][1]), "{:.0f}", "variant: process audit slot time, s")
v("av_pass_process", pct(AV_S.checks["process"][0]), "{:.0f}", "variant: process audit passes, %")
v("av_rej_process", sec(AV.cost_per_rejection["process"]), "{:.0f}", "variant: process audit, s per rejection")
v("av_saving", pct(AV.saving), "{:.1f}", "variant: saving with the tests first, %")

print("== Three denominators (Scenarios.RecoveryPoolDenominators)")
_dir = DN_S.classes["direct"]
_rec = DN_S.classes["recovery"]
v("dn_records", DN.records, "{:,.0f}", "records in the pool")
v("dn_rec_pct", pct(DN.record_share), "{:.0f}", "recovery share of records, %")
v("dn_stored_pct", pct(DN.stored_share), "{:.0f}", "recovery share of stored tokens, %")
v("dn_target_pct", pct(DN.target_share), "{:.0f}", "recovery share of selected targets, %")
v("dn_admitted_pct", pct(DN.admitted_stored_share), "{:.1f}", "recovery share of processed tokens, %")
v("dn_rec_len", _rec[1].to("token").magnitude, "{:,.0f}", "stored tokens per recovery record")
v("dn_dir_len", _dir[1].to("token").magnitude, "{:,.0f}", "stored tokens per direct record")
v("dn_ctx_rec", DN.context_per_target["recovery"], "{:.2f}", "recovery context per target")
v("dn_ctx_dir", DN.context_per_target["direct"], "{:.0f}", "direct context per target")

print("== Attempt-group contrast (Scenarios.EightAttemptGroupContrast)")
_p_lo, _p_mid, _p_hi = GC_S.pass_rates
v("gc_k", GC_S.attempts, "{:.0f}", "attempts per fixture")
v("gc_lo", _p_lo, "{:.2f}", "low pass rate")
v("gc_lo_pct", pct(_p_lo), "{:.0f}", "low pass rate, %")
v("gc_mid_pct", pct(_p_mid), "{:.0f}", "middle pass rate, %")
v("gc_hi_pct", pct(_p_hi), "{:.0f}", "high pass rate, %")
v("gc_mixed_lo", GC_S.mixed_fraction(_p_lo), "{:.2f}", "mixed fraction at the low rate")
v("gc_mixed_hi", GC_S.mixed_fraction(_p_hi), "{:.2f}", "mixed fraction at the high rate")
v("gc_mixed_mid", GC_S.mixed_fraction(_p_mid), "{:.2f}", "mixed fraction at the middle rate")
v("gc_yield_lo_pct", pct(GC_S.success_yield(_p_lo)), "{:.0f}", "a success at the low rate, %")
v("gc_attempts_lo", GC_S.attempts_per_success(_p_lo), "{:.0f}", "attempts per harvested success")

print("== Staging burst (Scenarios.VerifierTimeoutStagingBurst)")
_store_gb = ST_S.store.to("GB").magnitude
v("st_rollouts", ST_S.rollouts, "{:,.0f}", "concurrent rollouts")
v("st_rollout_s", sec(ST_S.rollout_time), "{:.0f}", "seconds per rollout")
v("st_slots", ST_S.slots, "{:.0f}", "verifier slots")
v("st_timeout", sec(ST_S.timeout), "{:.0f}", "timeout during the burst, s")
v("st_ceiling_mb", ST_S.record_ceiling.to("MB").magnitude, "{:.0f}", "record ceiling, MB")
v("st_store_gb", _store_gb, "{:.0f}", "staging store, GB")
v("st_react_s", sec(ST_S.reaction), "{:.0f}", "controller delay, s")
v("st_arrival", ST.arrival_rate.to("1/s").magnitude, "{:.1f}", "arrivals, records/s")
v("st_capacity", ST.capacity.to("1/s").magnitude, "{:.1f}", "nominal capacity, records/s")
v("st_util", ST.utilization, "{:.2f}", "utilization")
v("st_burst", ST.burst_capacity.to("1/s").magnitude, "{:.1f}", "burst capacity, records/s")
v("st_net", ST.net_growth.to("1/s").magnitude, "{:.1f}", "net growth, records/s")
v("st_growth", ST.growth.to("MB/s").magnitude, "{:.1f}", "growth, MB/s")
v("st_growth_r", ST.growth.to("MB/s").magnitude, "{:.0f}", "the same, whole MB/s")
v("st_tfull", sec(ST.time_to_full), "{:.0f}", "time to full, s")
v("st_tfull_1", sec(ST.time_to_full), "{:.1f}", "the same, to a tenth")
v("st_tfull_min", ST.time_to_full.to("minute").magnitude, "{:.1f}", "the same, minutes")
v("st_inflight", ST.in_flight.to("GB").magnitude, "{:.1f}", "in flight at stop, GB")
v("st_react_gb", ST.reaction_growth.to("GB").magnitude, "{:.1f}", "growth during the delay, GB")
v("st_mark_pct", math.floor(pct(ST.high_watermark)), "{:.0f}", "stop dispatch at or below, % of the store")
v("st_mark_gb", ST.high_watermark * _store_gb, "{:.1f}", "the same bound, GB")

print("== Cost per emitted artifact (Scenarios.CurationCohortDispositions)")
_disp = CO_S.dispositions
v("co_attempts", CO_S.attempts, "{:.0f}", "attempts in the cohort")
v("co_cost", CO_S.cohort_cost.to("USD").magnitude, "{:.0f}", "cohort cost, USD")
v("co_emitted", _disp["selected_emitted"], "{:.0f}", "selected and emitted")
v("co_other", _disp["assessed_not_selected"], "{:.0f}", "assessed, not selected")
v("co_inelig", _disp["ineligible"], "{:.0f}", "ineligible for this use")
v("co_pending", _disp["pending"], "{:.0f}", "pending at the cutoff")
v("co_unknown", _disp["unknown"], "{:.0f}", "unknown at the cutoff")
v("co_yield_pct", pct(CO_S.emission_yield), "{:.0f}", "emission yield, %")
v("co_yield", CO_S.emission_yield, "{:.2f}", "the same, as a fraction")
v("co_attempt_cost", CO_S.cost_per_attempt.to("USD").magnitude, "{:.2f}", "cost per attempt, USD")
v("co_artifact", CO_S.cost_per_artifact().to("USD").magnitude, "{:.0f}", "cost per emitted artifact, USD")
v("co_dbl_pct", pct(CO_S.yield_multiple * CO_S.emission_yield), "{:.0f}", "doubled yield, %")
v("co_dbl_cost", CO_S.cost_per_artifact(CO_S.yield_multiple * CO_S.emission_yield).to("USD").magnitude, "{:.0f}",
  "cost per artifact at the doubled yield, USD")
v("co_not_pct", pct(1 - CO_S.emission_yield), "{:.0f}", "attempts that emit nothing, %")

if __name__ == "__main__":
    write(__file__)
