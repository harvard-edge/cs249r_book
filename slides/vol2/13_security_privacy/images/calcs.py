#!/usr/bin/env python3
"""Every number on the Lecture 13 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (registry and Literature) and the calculations
(Scenarios.*, each a scenario type over a physics equation); the chapter,
Security and Privacy, reads the same scenarios. This file only selects
values, converts units, and formats them; it types no numbers. Each value is
recorded with v() under the key the deck prints as \\val{key}; running this
file writes images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Literature, Scenarios = mlsysim.Literature, mlsysim.Scenarios
STEAL = Literature.ExtractionAttacks
POISON = Literature.DataPoisoning
DP = Literature.DifferentialPrivacy
INCIDENTS = Literature.SecurityIncidents
NETFLIX = Literature.NetflixPrize

EXPOSURE = Scenarios.CanaryVersusCutoverExposure      # recall from Lecture 12
FLEET = Scenarios.ReferenceJobAttackSurface           # the running case
SHAPING = Scenarios.ImageNetTop5RoundedScores         # output contract
BUDGET = Scenarios.ResNet50ApiExtractionBudget        # Napkin Math 13.1
LAION = Scenarios.LaionWebScalePoisoning
MEAN = Scenarios.ByzantineMeanOnReferenceJob
ISOLATION = Scenarios.SecureMultiTenantServing
PASSWORD = Scenarios.FiveBytePasswordPowerAnalysis
R50_FP32, R50_INT8 = Scenarios.ResNet50Fp32InSgxEnclave, Scenarios.ResNet50Int8InSgxEnclave
R18_INT8 = Scenarios.ResNet18Int8InSgxEnclave
ENCRYPTED = Scenarios.HealthMonitoringEncryptedInference  # Napkin Math 13.2
AUDIT = Scenarios.MembershipAuditAtEpsilonOne
SALARY = Scenarios.SalaryMeanLaplaceRelease           # Napkin Math 13.3
COMPOSE = Scenarios.TenReleasesBasicComposition
ACCOUNT = Scenarios.DpSgdAccountantComparison
RUN = Scenarios.Cifar10DpSgdRun


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def pct(x):
    return 100 * x


def mil(x):
    return x / 10 ** 6


def k(x):
    return x / 10 ** 3


print("== Recall: a canary against a cutover (Lecture 12, Napkin Math 12.5)")
v("canary_pct", pct(EXPOSURE.canary_share), "{:.0f}", "canary share, %")
v("cutover_pct", pct(EXPOSURE.full_share), "{:.0f}", "cutover share, %")
v("exposure_ratio", EXPOSURE.exposure_ratio, "{:.0f}", "exposure ratio")

print("== Reference fleet (Sec. 13.6)")
v("ref_gpus", FLEET.accelerators, "{:,.0f}", "GPUs")
v("ref_nodes", FLEET.nodes, "{:.0f}", "nodes")
v("ref_gpn", FLEET.accelerators_per_node, "{:.0f}", "GPUs per node")
v("grad_gb", m(FLEET.gradient_bytes, "GB"), "{:.0f}", "FP16 gradients per step, GB")
v("ckpt_tb", m(FLEET.checkpoint_bytes, "TB"), "{:.2f}", "checkpoint, TB")
v("weights_gb", m(FLEET.weight_bytes, "GB"), "{:.0f}", "FP16 weights, GB")

print("== Incidents (Sec. 13.3)")
v("stuxnet_zd", float(INCIDENTS.StuxnetZeroDayExploits), "{:.0f}", "Stuxnet zero-day exploits")
v("stuxnet_zd_word", float(INCIDENTS.StuxnetZeroDayExploits), "words", "the same, in words")
v("mirai_peak", float(INCIDENTS.MiraiPeakInfections), "{:,.0f}", "Mirai peak infections")

print("== Production-API extraction (Carlini et al. 2024; table 4)")
v("steal_small_usd", float(STEAL.SmallModelMatrixCostUpperBoundUsd), "{:.0f}", "ada and babbage full matrix, under $")
v("steal_gpt35_usd", float(STEAL.Gpt35MatrixCostEstimateUsd), "{:,.0f}", "gpt-3.5-turbo full matrix, estimated under $")
v("steal_ada_rms", float(STEAL.AdaProjectionRms), "{:.4f}", "ada RMS error")
v("steal_babbage_rms", float(STEAL.BabbageProjectionRms), "{:.4f}", "babbage RMS error")
v("steal_b002_width", float(STEAL.Babbage002HiddenWidth), "{:,.0f}", "babbage-002 width")
v("steal_ada_q_m", mil(float(STEAL.AdaQueriesUpperBound)), "{:.0f}", "ada queries, under, millions")
v("steal_babbage_q_m", mil(float(STEAL.BabbageQueriesUpperBound)), "{:.0f}", "babbage queries, under, millions")
v("steal_gpt35_q_m", mil(float(STEAL.Gpt35QueriesUpperBound)), "{:.0f}", "gpt-3.5-turbo queries, under, millions")

print("== Output contract (Sec. 13.5.1)")
v("shape_classes", SHAPING.class_count, "{:,.0f}", "class scores")
v("shape_topk", SHAPING.top_k, "{:.0f}", "top k returned")
v("shape_fewer", SHAPING.scores_reduction, "{:.0f}", "times fewer scores per answer")
v("shape_removed_pct", pct(SHAPING.entries_removed), "{:.1f}", "entries removed, %")
v("shape_from", SHAPING.from_decimals, "{:.0f}", "decimal places before rounding")
v("shape_to", SHAPING.to_decimals, "{:.0f}", "decimal places after rounding")
v("shape_bits", SHAPING.bits_removed, "{:.1f}", "bits removed per class")
v("shape_digits", SHAPING.decimal_digits_removed, "{:.0f}", "decimal digits removed")
v("shape_bits_per_digit", SHAPING.bits_removed / SHAPING.decimal_digits_removed, "{:.1f}", "bits per decimal digit")

print("== Napkin Math 13.1: daily quota against a lifetime budget (scenario)")
v("ext_train_k", k(m(BUDGET.training_cost, "USD")), "{:.0f}", "training cost, $K")
v("ext_train", m(BUDGET.training_cost, "USD"), "{:,.0f}", "training cost, $")
v("ext_queries_m", mil(BUDGET.extraction_queries), "{:.0f}", "queries for the copy, millions")
v("ext_fid_pct", pct(BUDGET.target_fidelity), "{:.0f}", "copy fidelity, %")
v("ext_tier_usd", m(BUDGET.tier_price, "USD"), "{:.0f}", "paid tier, $ a month")
v("ext_quota_k", k(BUDGET.tier_daily_quota), "{:.0f}", "paid-tier quota, K a day")
v("ext_quota", BUDGET.tier_daily_quota, "{:,.0f}", "paid-tier quota a day")
v("ext_price", m(BUDGET.price_per_query, "USD"), "{:.3f}", "price per query, $")
v("ext_days", BUDGET.collection_days, "{:.0f}", "days of collection on the paid tier")
v("ext_period", m(BUDGET.billing_period, "day"), "{:.0f}", "days per billing month")
v("ext_tier_cost", m(BUDGET.full_set_tier, "USD"), "{:,.0f}", "paid-tier bill for the full set, $")
v("ext_payg", m(BUDGET.full_set_pay_per_query, "USD"), "{:,.0f}", "pay-per-query bill for the full set, $")
v("ext_allow_k", k(BUDGET.lifetime_allowance), "{:.0f}", "lifetime allowance, K queries")
v("ext_allow", BUDGET.lifetime_allowance, "{:,.0f}", "lifetime allowance, queries")
v("ext_allow_days", BUDGET.tier_days(BUDGET.lifetime_allowance), "{:.0f}", "days to use the allowance")
v("ext_allow_fees", m(BUDGET.allowance_fees, "USD"), "{:.0f}", "tier fees during the allowance, $")
v("ext_overage", m(BUDGET.overage_cost, "USD"), "{:,.0f}", "overage billed, $")
v("ext_overage_q_m", mil(BUDGET.extraction_queries - BUDGET.lifetime_allowance), "{:.1f}", "queries past the allowance, millions")
v("ext_life", m(BUDGET.full_set_lifetime, "USD"), "{:,.0f}", "lifetime-budget bill, $")
v("ext_tier_share", BUDGET.tier_share_of_training, "{:.2f}", "paid-tier bill / training cost")

print("== Poisoning (Shan et al. 2024; Carlini et al. 2024, S&P)")
v("ns_low", float(POISON.NightshadeDoseLow), "{:.0f}", "Nightshade dose, low")
v("ns_mid", float(POISON.NightshadeDoseMid), "{:.0f}", "Nightshade dose, middle")
v("ns_high", float(POISON.NightshadeDoseHigh), "{:.0f}", "Nightshade dose, high")
v("laion_pct", pct(LAION.poisoned_fraction), "{:.2f}", "share poisonable, %")
v("laion_images", LAION.poisoned_items, "{:,.0f}", "LAION-400M images in that share")
v("laion_usd", m(LAION.cost, "USD"), "{:.0f}", "cost, $")

print("== Breakdown point and robust gather (Sec. 13.6.1)")
v("bd_workers", MEAN.workers, "{:.0f}", "workers")
v("bd_honest_n", MEAN.workers - 1, "{:.0f}", "honest workers")
v("bd_honest", MEAN.honest_value, "{:.1f}", "honest value")
v("bd_attack", MEAN.attacker_value, "{:,.0f}", "attacker's value")
v("bd_mean", MEAN.mean(), "{:.0f}", "mean of the 64")
v("dp", FLEET.dp_degree, "{:.0f}", "data-parallel replicas")
v("dp_gpus", FLEET.accelerators_per_replica, "{:.0f}", "GPUs per replica")
v("dp_nodes_per_replica_word", FLEET.accelerators_per_replica / FLEET.accelerators_per_node, "words",
  "nodes per replica, in words")
v("ring", FLEET.allreduce_factor, "{:.2f}", "ring AllReduce bytes per worker (x S)")
v("gather", FLEET.gather_factor, "{:.0f}", "gather bytes per worker (x S)")
v("gather_ratio", FLEET.gather_traffic_ratio, "{:.0f}", "gather over ring, times")
v("mt_base", m(ISOLATION.baseline_rate, "1/s"), "{:,.0f}", "matched serving, tokens/s")
v("mt_ctrl", m(ISOLATION.controlled_rate, "1/s"), "{:.0f}", "with isolation, tokens/s")
v("mt_pct", pct(ISOLATION.overhead), "{:.0f}", "throughput lost, %")

print("== Side channel (Sec. 13.7.4) and protected memory (Sec. 13.8.1, Napkin Math 13.2)")
v("sc_symbols", PASSWORD.symbols, "{:.0f}", "values per byte")
v("sc_bytes", PASSWORD.secret_bytes, "{:.0f}", "password bytes")
v("sc_exp", math.floor(math.log10(PASSWORD.brute_force_guesses)), "{:.0f}", "brute-force guesses, power of ten")
v("epc_mib", m(R50_FP32.capacity, "MiB"), "{:.0f}", "enclave page cache, MiB")
v("epc_mb", m(R50_FP32.capacity, "MB"), "{:.1f}", "the same, MB")
v("r50_fp32_mb", m(R50_FP32.weight_bytes, "MB"), "{:.1f}", "ResNet-50 FP32 weights, MB")
v("r50_fp32_pct", pct(R50_FP32.capacity_share), "{:.0f}", "share of the EPC, %")
v("r50_int8_mb", m(R50_INT8.weight_bytes, "MB"), "{:.1f}", "ResNet-50 INT8 weights, MB")
v("r18_int8_mb", m(R18_INT8.weight_bytes, "MB"), "{:.1f}", "ResNet-18 INT8 weights, MB")
v("fhe_mult", ENCRYPTED.fhe_multiplier, "{:,.0f}", "FHE overhead, times (assumed)")
v("fhe_base_ms", m(ENCRYPTED.plaintext_latency, "ms"), "{:.0f}", "plaintext inference, ms")
v("fhe_aes_ms", m(ENCRYPTED.encryption_overhead, "ms"), "{:.1f}", "AES adds, ms")
v("fhe_aes_total_ms", m(ENCRYPTED.encrypted_transport_latency, "ms"), "{:.1f}", "with AES, ms")
v("fhe_s", m(ENCRYPTED.fhe_latency, "second"), "{:.0f}", "under FHE, s")

print("== Memorization (Carlini et al. 2021), audit ceiling, deanonymization")
v("gpt2_gen", float(STEAL.Gpt2SamplesGenerated), "{:,.0f}", "GPT-2 samples generated")
v("gpt2_insp", float(STEAL.Gpt2SamplesInspected), "{:,.0f}", "samples inspected")
v("gpt2_mem", float(STEAL.Gpt2SamplesMemorized), "{:.0f}", "confirmed memorized")
v("audit_eps", AUDIT.epsilon, "{:.0f}", "audit figure epsilon")
v("nf_ratings", float(NETFLIX.DeanonymizationRatings), "{:.0f}", "ratings with approximate dates")
v("nf_pct", float(NETFLIX.DeanonymizationRecordsPct), "{:.0f}", "records identified, %")

print("== Napkin Math 13.3: the salary mean")
v("sal_n", SALARY.group_size, "{:,.0f}", "people")
v("sal_range", m(SALARY.value_range, "USD"), "{:,.0f}", "salary ceiling, $")
v("sal_eps", SALARY.epsilon, "{:.0f}", "epsilon")
v("sal_sens", m(SALARY.sensitivity, "USD"), "{:,.0f}", "sensitivity of the sum, $")
v("sal_scale", m(SALARY.noise_scale, "USD"), "{:,.0f}", "Laplace scale on the sum, $")
v("sal_err", m(SALARY.mean_error(), "USD"), "{:.0f}", "expected error of the mean, $")
v("sal_small_n", SALARY.small_group_size, "{:.0f}", "small group, people")
v("sal_small_err", m(SALARY.small_group_error, "USD"), "{:,.0f}", "expected error, small group, $")
v("sal_wrong", m(SALARY.mean_error(), "USD") / SALARY.group_size, "{:.1f}", "common error: divide by the count twice, $")

print("== Composition and the accountant (Sec. 13.9.2; Abadi et al. 2016)")
v("comp_n_word", COMPOSE.releases, "words", "releases, in words")
v("comp_eps", COMPOSE.epsilon_per_release, "{:.0f}", "epsilon per release")
v("comp_total", COMPOSE.total_epsilon, "{:.0f}", "basic-composition bound")
short, long_ = sorted(ACCOUNT.accountant)
v("ab_q", ACCOUNT.sampling_rate, "{:.2f}", "sampling rate")
v("ab_sigma", ACCOUNT.noise_multiplier, "{:.0f}", "noise multiplier")
v("ab_delta_exp", -math.log10(ACCOUNT.delta), "{:.0f}", "delta, negative power of ten")
v("ab_short", short, "{:.0f}", "shorter run, epochs")
v("ab_long", long_, "{:.0f}", "longer run, epochs")
v("ab_acc_short", ACCOUNT.accountant[short], "{:.2f}", "accountant epsilon, shorter run")
v("ab_acc_long", ACCOUNT.accountant[long_], "{:.2f}", "accountant epsilon, longer run")
v("ab_strong_short", ACCOUNT.strong_composition[short], "{:.2f}", "strong composition, shorter run")
v("ab_strong_long", ACCOUNT.strong_composition[long_], "{:.2f}", "strong composition, longer run")
v("ab_tight_short", ACCOUNT.tightening(short), "{:.1f}", "tighter accounting, shorter run, times")
v("ab_tight_long", ACCOUNT.tightening(long_), "{:.1f}", "tighter accounting, longer run, times")
v("run_epochs", RUN.epochs, "{:.0f}", "CIFAR-10 run, epochs")
v("run_eps", RUN.target_epsilon, "{:.0f}", "target epsilon")
v("run_eps_code", RUN.target_epsilon, "{:.1f}", "target epsilon, as code")
v("run_delta_exp", -math.log10(RUN.delta), "{:.0f}", "delta, negative power of ten")
v("run_clip", RUN.clip_norm, "{:.0f}", "clipping norm C")
v("run_clip_code", RUN.clip_norm, "{:.1f}", "clipping norm, as code")
mnist = float(DP.MnistNonprivateAccuracyPct)
cifar = float(DP.Cifar10NonprivateAccuracyPct)
v("mnist_base_pct", mnist, "{:.1f}", "MNIST nonprivate baseline, %")
# Same gap the chapter computes: the largest shortfall from the MNIST baseline
# among the plotted MNIST points at single-digit epsilon (1 <= epsilon < 10),
# Bu et al. 2020 (Table 1, CLT epsilon) and Abadi et al. 2016 (Section 5.2).
mnist_points = [(r["clt_epsilon"], r["test_accuracy_pct"]) for r in Literature.Series.GaussianDpMnistNoisySgd.entries]
mnist_points += [(r["epsilon"], r["accuracy_pct"]) for r in Literature.Series.AbadiDpSgdAccuracy.entries
                 if r["dataset"] == "MNIST"]
mnist_gap = mnist - min(acc for eps, acc in mnist_points if 1 <= eps < 10)
v("mnist_gap_pts", mnist_gap, "{:.1f}", "largest MNIST gap at single-digit epsilon, points")
v("cifar_base_pct", cifar, "{:.1f}", "CIFAR-10 nonprivate reference (WRN-16-4), %")
v("cifar_gap_lo", cifar - float(DP.Cifar10DeAccuracyAtEps8Pct), "{:.0f}", "CIFAR-10 gap, De et al., points")
v("cifar_gap_hi", cifar - float(DP.Cifar10AbadiAccuracyAtEps8Pct), "{:.0f}", "CIFAR-10 gap, Abadi et al., points")
v("cifar_eps", float(DP.Cifar10ComparisonEpsilon), "{:.0f}", "CIFAR-10 comparison epsilon")

if __name__ == "__main__":
    write(__file__)
