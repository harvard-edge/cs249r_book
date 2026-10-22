#!/usr/bin/env python3
"""Every number on the Lecture 9 (Data Selection) slides and notes, read from MLSysIM.

MLSysIM holds the inputs (Models, Hardware, Datasets, Literature, and the
data-selection Scenarios.*) and the calculations; the Data Selection chapter
reads the same scenarios. This file only selects values, converts units, and
formats them; it types no numbers. Each value is recorded with v() under the
key the deck prints as \\val{key}; running this file writes images/numbers.tex.
make_figs.py imports this module, so figures and slides share one computation.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402

Literature, Models, Scenarios = mlsysim.Literature, mlsysim.Models, mlsysim.Scenarios
DSR, CHIN = Literature.DataSelectionResults, Literature.Chinchilla

CORPUS = Scenarios.PretrainingCorpusGrowth
GPT2, GPT2_ONE = Scenarios.Gpt2XlWebTextCorpus, Scenarios.Gpt2XlRunOnOneH100
DOUBLING = Scenarios.ChinchillaComputeDoubling
DEDUP, BANDS = Scenarios.Gpt2WebTextNearDuplicates, Scenarios.FineWebMinHashBanding
ICR = Scenarios.ResNet50HalfImageNetIcr
CORESET, PRACTICE = Scenarios.ResNet50TenPercentCoresetH100, Scenarios.ResNet50ProxyTrainedCoreset
EPOCH, ECHO = Scenarios.ResNet50HalfCoresetEpochH100, Scenarios.ResNet50EchoingH100
CHECK, PITFALL = Scenarios.ProductionModelScoringCheck, Scenarios.ExpensiveSelectionPitfall
COVER, CURRIC = Scenarios.RareClassTenPercentCoreset, Scenarios.Cifar10LinearCurriculum
ROUNDS, SCANS = Scenarios.ActiveLearningTenRounds, Scenarios.MedicalScanActiveLearning
SSL, FOUNDATION = Scenarios.FixMatchCifar10Cost, Scenarios.FoundationModelTenTasks
BILL, PIPELINE = Scenarios.ImageNetDataCostBreakdown, Scenarios.MinHashPipelineAmortization
SCALING = Scenarios.TenfoldDataFourPoints
CIFAR10 = mlsysim.Datasets.CIFAR10


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def lit(entry):
    """A Literature value as a plain number."""
    return float(entry)


def pct(x):
    """A fraction as percent."""
    return 100 * x


def sci(key, x, label):
    """Record x as a one-decimal mantissa (key_m) and a power of ten (key_e)."""
    e = math.floor(math.log10(abs(x)))
    v(f"{key}_m", x / 10 ** e, "{:.1f}", f"{label}, mantissa")
    v(f"{key}_e", e, "{:.0f}", f"{label}, power of ten")


print("== The data wall (local CSV series, Scenarios.PretrainingCorpusGrowth)")
corpora = load_series("vol1/04_data_engineering/data/pretraining_tokens.csv").entries
stock = {r["bound"]: r["tokens"] for r in load_series("vol1/04_data_engineering/data/public_text_stock.csv").entries}
v("corpus_growth", CORPUS.growth, "{:.1f}", "pre-training corpus growth per year")
v("first_corpus_b", corpora[0]["tokens"] / 10 ** 9, "{:.0f}", "GPT-3 training tokens, billions")
v("first_corpus_year", int(corpora[0]["date"].split("-")[0]), "{:.0f}", "GPT-3 year")
v("last_corpus_t", corpora[-1]["tokens"] / 10 ** 12, "{:.0f}", "Qwen3 training tokens, trillions")
v("last_corpus_year", int(corpora[-1]["date"].split("-")[0]), "{:.0f}", "Qwen3 year")
v("stock_low_t", stock["low"] / 10 ** 12, "{:.0f}", "public text stock, low, trillions")
v("stock_median_t", stock["median"] / 10 ** 12, "{:.0f}", "public text stock, median, trillions")
v("stock_high_q", stock["high"] / 10 ** 15, "{:.0f}", "public text stock, high, quadrillions")
v("stock_year", round(CORPUS.crossing(stock["median"])), "{:.0f}", "year the fit reaches the median stock")

print("== GPT-2 XL on WebText (Scenarios.Gpt2XlWebTextCorpus)")
v("gpt2_tokens_b", round(GPT2.tokens_processed.magnitude, -9) / 10 ** 9, "{:.0f}", "tokens processed, billions")
v("webtext_tokens_b", GPT2.unique_tokens.magnitude / 10 ** 9, "{:.0f}", "WebText unique tokens, billions")
v("webtext_gb", m(GPT2.corpus_bytes, "GB"), "{:.0f}", "WebText, GB")
v("gpt2_passes", GPT2.passes, "{:.0f}", "passes over WebText")
v("optimal_tokens_b", GPT2.optimal_tokens.magnitude / 10 ** 9, "{:.0f}", "compute-optimal tokens, billions")
v("gpt2_params_b", GPT2.parameters / 10 ** 9, "{:.1f}", "GPT-2 XL parameters, billions")
sci("gpt2_flops", m(GPT2.training_flops, "flop"), "original run, FLOPs")
v("flops_per_pt", GPT2.flops_per_param_token, "{:.0f}", "training FLOPs per parameter per token")
v("flops_per_pt_word", GPT2.flops_per_param_token, "words", "the same, in prose")
v("node_gpus_word", GPT2.accelerators, "words", "accelerators in one machine, in prose")
v("mfu_pct", pct(GPT2.utilization), "{:.0f}", "model FLOPs utilization, %")
v("node_days", m(GPT2.run_time, "day"), "{:.1f}", "eight H100s, days")
v("node_kwh", round(m(GPT2.run_energy, "kWh"), -1), "{:.0f}", "eight H100s, kWh (to the ten, as the chapter rounds)")
v("node_hours", m(GPT2.run_time, "hour"), "{:.0f}", "eight H100s, hours")
v("one_h100_days", m(GPT2_ONE.training_time, "day"), "{:.0f}", "one H100, days")

print("== Model collapse (Shumailov et al. 2024)")
v("shum_ppl", lit(DSR.ShumailovRealDataPerplexity), "{:.0f}", "real-data model, mean perplexity")
v("shum_loss_lo", lit(DSR.ShumailovGeneratedLossLowPts), "{:.0f}", "perplexity points lost, low")
v("shum_loss_hi", lit(DSR.ShumailovGeneratedLossHighPts), "{:.0f}", "perplexity points lost, high")
v("shum_real_pct", lit(DSR.ShumailovPreservedRealPct), "{:.0f}", "real data kept each generation, %")

print("== Value: learning per FLOP (Scenarios.ResNet50HalfImageNetIcr)")
v("imagenet_m", ICR.examples / 10 ** 6, "{:.2f}", "ImageNet training images, millions")
v("gflop_per_image", m(ICR.training_flops_per_sample, "GFLOP"), "{:.1f}", "training FLOPs per image, GFLOP")
v("icr_gain_full", ICR.gain_full, "{:.0f}", "epoch-1 gain, all data, points")
v("icr_gain_subset", ICR.gain_subset, "{:.1f}", "epoch-1 gain, half, points")
v("icr_gain_share", pct(ICR.gain_subset / ICR.gain_full), "{:.0f}", "half's share of the gain, %")
sci("epoch_flops", m(ICR.epoch_flops, "flop"), "one epoch, FLOPs")
sci("half_flops", m(ICR.subset_flops, "flop"), "half an epoch, FLOPs")
sci("icr_full", ICR.icr_full, "ICR, all data, per FLOP")
sci("icr_half", ICR.icr_subset, "ICR, half, per FLOP")
v("icr_ratio", ICR.icr_ratio, "{:.1f}", "half / all, learning per FLOP")
v("paul_epoch", lit(DSR.PaulEl2nScoreEpoch), "{:.0f}", "EL2N scored at epoch")
v("sorscher_discard_pct", lit(DSR.SorscherImageNetDiscardPct), "{:.0f}", "ImageNet discarded without loss, %")
v("sorscher_kept_pct", 100 - lit(DSR.SorscherImageNetDiscardPct), "{:.0f}", "the same, kept, %")

print("== Tokens per parameter (Models, Scenarios.Gpt2XlWebTextCorpus)")
gpt3, llama2 = Models.Language.GPT3, Models.Language.Llama2_70B
v("gpt3_tokens_b", m(gpt3.training_tokens, "count") / 10 ** 9, "{:.0f}", "GPT-3 training tokens, billions")
v("gpt3_params_b", m(gpt3.parameters, "param") / 10 ** 9, "{:.0f}", "GPT-3 parameters, billions")
v("gpt3_tpp", m(gpt3.training_tokens, "count") / m(gpt3.parameters, "param"), "{:.1f}", "GPT-3 tokens per parameter")
v("llama2_tpp", m(llama2.training_tokens, "count") / m(llama2.parameters, "param"), "{:.1f}",
  "Llama 2 70B tokens per parameter")
v("gpt2_unique_tpp", GPT2.unique_tokens_per_param, "{:.1f}", "GPT-2 XL unique tokens per parameter")
v("gpt2_trained_tpp", GPT2.trained_tokens_per_param, "{:.0f}", "GPT-2 XL training tokens per parameter")
v("tpp_ref", GPT2.tokens_per_param, "{:.0f}", "compute-optimal tokens per parameter")
v("processed_over_optimal", (GPT2.tokens_processed / GPT2.optimal_tokens).to("").magnitude, "{:.1f}",
  "processed / compute-optimal tokens")

print("== The compute-optimal frontier (Literature.Chinchilla, Scenarios.ChinchillaComputeDoubling)")
v("alloc_exp", DOUBLING.exponent, "{:.2f}", "approach 1 allocation exponent")
v("alloc_exp_short", DOUBLING.exponent, "{:.1f}", "the same, one decimal")
v("chin_models", lit(CHIN.ModelsTrained), "{:.0f}", "models trained, over")
v("gopher_params_b", lit(CHIN.GopherParams) / 10 ** 9, "{:.0f}", "Gopher parameters, billions")
v("gopher_tokens_b", lit(CHIN.GopherTokens) / 10 ** 9, "{:.0f}", "Gopher tokens, billions")
v("chin_params_b", lit(CHIN.ChinchillaParams) / 10 ** 9, "{:.0f}", "Chinchilla parameters, billions")
v("chin_tokens_t", lit(CHIN.ChinchillaTokens) / 10 ** 12, "{:.1f}", "Chinchilla tokens, trillions")
v("chin_token_mult", lit(CHIN.ChinchillaTokens) / lit(CHIN.GopherTokens), "{:.1f}", "Chinchilla / Gopher tokens")
v("chin_tpp", lit(CHIN.ChinchillaTokens) / lit(CHIN.ChinchillaParams), "{:.0f}", "Chinchilla tokens per parameter")
v("doubling_pct", pct(DOUBLING.growth), "{:.0f}", "more tokens when compute doubles, %")
v("c_over_n2", GPT2.flops_per_param_token * GPT2.tokens_per_param, "{:.0f}", "C / N^2 at D = 20 N")
n_opt, d_opt = GPT2.compute_optimal
v("gpt2_nopt_b", n_opt / 10 ** 9, "{:.1f}", "compute-optimal parameters, billions")
v("gpt2_dopt_b", d_opt / 10 ** 9, "{:.0f}", "compute-optimal tokens, billions")
v("dopt_over_webtext", d_opt / GPT2.unique_tokens.magnitude, "{:.1f}", "D_opt / WebText unique tokens")

print("== The selection inequality (Scenarios.ResNet50TenPercentCoresetH100)")
v("util_pct", pct(CORESET.utilization), "{:.0f}", "sustained share of H100 peak, %")
v("keep10_pct", pct(CORESET.keep_fraction), "{:.0f}", "coreset, %")
v("train_epochs", CORESET.epochs, "{:.0f}", "training epochs")
v("base_min", m(CORESET.full_time, "minute"), "{:.0f}", "full training, min")
v("score_s", m(CORESET.score_time(), "second"), "{:.0f}", "ResNet-50 scores the pool, s")
v("score_proxy_s", m(CORESET.score_time(CORESET.proxy), "second"), "{:.0f}", "ResNet-18 scores the pool, s")
v("score_diff_s", m(CORESET.score_time() - CORESET.score_time(CORESET.proxy), "second"), "{:.0f}",
  "proxy scoring saves, s")
v("save_a_min", m(CORESET.saving(), "minute"), "{:.1f}", "saving, target scores once, min")
v("save_a_pct", pct(CORESET.saving_fraction()), "{:.1f}", "the same, %")
v("save_b_min", m(CORESET.saving(CORESET.proxy), "minute"), "{:.1f}", "saving, proxy scores once, min")
v("save_b_pct", pct(CORESET.saving_fraction(CORESET.proxy)), "{:.1f}", "the same, %")
v("rescore_min", m(CORESET.rescoring_time, "minute"), "{:.0f}", "rescoring every epoch, min")
v("subset_min", m(CORESET.subset_time, "minute"), "{:.0f}", "subset training, min")
v("rescore_ratio", CORESET.rescoring_over_training, "{:.1f}", "rescoring / subset training")
v("rescore_save_pct", pct(CORESET.rescoring_saving_fraction), "{:.0f}", "saving with rescoring, %")
v("rescore_lost_pct", pct(1 - CORESET.rescoring_saving_fraction / CORESET.saving_fraction()), "{:.0f}",
  "share of the one-pass saving lost, %")

print("== Estimate: 12 hours of scoring (Scenarios.ProductionModelScoringCheck)")
v("est_pool", CHECK.pool, "{:,.0f}", "pool, images")
v("est_kept", CHECK.kept, "{:,.0f}", "kept, images")
v("est_cut_pct", pct(1 - CHECK.kept / CHECK.pool), "{:.0f}", "cut, %")
v("est_full_h", m(CHECK.full_time, "hour"), "{:.0f}", "full training, h")
v("est_subset_h", m(CHECK.subset_time, "hour"), "{:.0f}", "subset training, h")
v("est_score_h", m(CHECK.selection_time, "hour"), "{:.0f}", "production-model scoring, h")
v("est_total_h", m(CHECK.total_time, "hour"), "{:.0f}", "scoring + subset, h")
v("est_loss_h", -m(CHECK.net_saving, "hour"), "{:.0f}", "hours lost")
v("est_breakeven_h", m(CHECK.breakeven_selection, "hour"), "{:.0f}", "scoring must fall below, h")

print("== Deduplication and MinHash (Scenarios.Gpt2WebTextNearDuplicates, FineWebMinHashBanding)")
v("lee_repeats", lit(DSR.LeeC4SentenceRepeats), "{:,.0f}", "repeats of one C4 sentence")
v("lee_words", lit(DSR.LeeC4SentenceWords), "{:.0f}", "words in that sentence")
v("minhash_hashes", DEDUP.signature_hashes, "{:.0f}", "MinHash values per document")
v("webtext_docs_m", DEDUP.documents / 10 ** 6, "{:.0f}", "WebText documents, millions")
sci("all_pairs", DEDUP.all_pairs, "all-pairs comparisons")
v("read_s", m(DEDUP.read_time, "second"), "{:.0f}", "one read of WebText, s")
v("sig_gb", m(DEDUP.signature_bytes, "GB"), "{:.1f}", "MinHash signatures, GB")
v("fw_tokens_t", lit(DSR.FineWebTokensT), "{:.0f}", "FineWeb, trillions of tokens")
v("fw_snapshots", lit(DSR.FineWebSnapshots), "{:.0f}", "Common Crawl snapshots")
v("fw_bands", BANDS.bands, "{:.0f}", "MinHash bands")
v("fw_rows", BANDS.rows, "{:.0f}", "hashes per band")
v("fw_shingle", BANDS.shingle_words, "{:.0f}", "shingle words")
v("fw_hashes", BANDS.hashes, "{:.0f}", "hashes per document")
v("fw_thresh", BANDS.threshold, "{:.2f}", "candidate threshold")
v("fw_thresh_pct", pct(BANDS.threshold), "{:.0f}", "the same, %")
probe_lo, probe_hi = BANDS.probe_similarities
v("fw_probe_lo", probe_lo, "{:.1f}", "a dissimilar pair's Jaccard similarity")
v("fw_probe_hi", probe_hi, "{:.1f}", "a near-duplicate pair's Jaccard similarity")
v("fw_p_lo_pct", pct(BANDS.candidate_probability(probe_lo)), "{:.1f}", "candidate chance at the low probe, %")
v("fw_p_hi_pct", pct(BANDS.candidate_probability(probe_hi)), "{:.1f}", "candidate chance at the high probe, %")
v("lee_c4_pct", pct(DEDUP.duplicate_shares["C4"]), "{:.2f}", "C4 near-duplicates, %")
v("lee_rn_pct", pct(DEDUP.duplicate_shares["RealNews"]), "{:.2f}", "RealNews near-duplicates, %")
v("dedup_c4_h", m(DEDUP.time_saved("C4"), "hour"), "{:.0f}", "C4 share of the run, h")
v("dedup_rn_h", m(DEDUP.time_saved("RealNews"), "hour"), "{:.0f}", "RealNews share of the run, h")
v("dedup_c4_kwh", m(DEDUP.energy_saved("C4"), "kWh"), "{:.0f}", "C4 share of the run, kWh")
v("dedup_rn_kwh", round(m(DEDUP.energy_saved("RealNews"), "kWh"), -1), "{:.0f}",
  "RealNews share of the run, kWh (to the ten, as the chapter rounds)")
v("lee_mem_word", lit(DSR.LeeMemorizationReduction), "words", "less memorized text, times, in prose")
v("lee_overlap_pct", lit(DSR.LeeValidationOverlapPct), "{:.0f}", "validation examples also in training, over %")

print("== Pruning with a proxy and a coverage floor (Scenarios.ResNet50ProxyTrainedCoreset, RareClassTenPercentCoreset)")
v("proxy_epochs", PRACTICE.proxy_epochs, "{:.0f}", "proxy training epochs")
v("std_epochs", EPOCH.epochs, "{:.0f}", "standard ImageNet schedule, epochs")
v("cov_pool", COVER.pool, "{:,.0f}", "pool, examples")
v("cov_pool_m", COVER.pool / 10 ** 6, "{:.0f}", "pool, millions")
v("cov_rare_pct", pct(COVER.rare_share), "{:.1f}", "rare class, %")
v("cov_keep_pct", pct(COVER.keep_fraction), "{:.0f}", "coreset, %")
v("cov_floor", COVER.floor, "{:.0f}", "coverage floor")
v("cov_rare", COVER.rare, "{:,.0f}", "rare examples in the pool")
v("cov_kept", COVER.expected_kept, "{:.0f}", "kept by an unstratified coreset")
v("cov_extra", COVER.shortfall, "{:.0f}", "extra rare examples the floor adds")

print("== Curriculum, proxies, and active learning (Scenarios.Cifar10LinearCurriculum, ActiveLearningTenRounds, ...)")
v("warm_w", CURRIC.warmup_epochs, "{:.0f}", "warmup epochs W")
v("warm_served", CURRIC.warmup_samples, "{:.1f}", "warmup serves, dataset passes")
v("cur_epochs", CURRIC.curriculum_epochs, "{:.0f}", "curriculum epochs to target")
v("cur_base_epochs", CURRIC.baseline_epochs, "{:.0f}", "shuffled epochs to target")
v("svp_speedup", lit(DSR.ColemanProxySpeedup), "{:.0f}", "proxy speedup, times")
v("svp_speedup_word", lit(DSR.ColemanProxySpeedup), "words", "the same, in prose")
v("svp_err_pct", lit(DSR.ColemanActiveErrorGapPct), "{:.1f}", "final-error gap, %")
v("svp_end2end", lit(DSR.ColemanEndToEndSpeedup), "{:.1f}", "end-to-end speedup, times")
v("svp_removed_pct", lit(DSR.ColemanCoresetRemovedPct), "{:.0f}", "CIFAR10 removed by the proxy, up to %")
v("svp_datasets_word", lit(DSR.ColemanDatasets), "words", "datasets evaluated, in prose")
v("al_price", m(ROUNDS.label_price, "USD"), "{:.0f}", "label price, $")
v("al_start", ROUNDS.initial_labels, "{:,.0f}", "starting labels")
v("al_per_round", ROUNDS.labels_per_round, "{:.0f}", "labels per round")
v("al_rounds", ROUNDS.rounds, "{:.0f}", "rounds")
v("al_round_usd", m(ROUNDS.round_scoring_cost, "USD"), "{:.0f}", "scoring per round, $")
v("al_random", ROUNDS.random_labels, "{:,.0f}", "random labels needed")
v("al_active", ROUNDS.active_labels, "{:,.0f}", "active labels after the rounds")
v("al_random_usd", m(ROUNDS.random_cost, "USD"), "{:,.0f}", "random labeling, $")
v("al_active_usd", m(ROUNDS.active_cost, "USD"), "{:,.0f}", "active learning, $")
v("al_roi", pct(ROUNDS.roi), "{:.1f}", "return, %")
v("al_turnaround_d", m(ROUNDS.label_turnaround, "day"), "{:.0f}", "annotation delay per round, days")
v("hosp_scans_m", m(SCANS.pool, "count") / 10 ** 6, "{:.0f}", "scans, millions")
v("hosp_price", m(SCANS.label_price, "USD/count"), "{:.0f}", "label, $")
v("hosp_all_m", m(SCANS.full_cost, "USD") / 10 ** 6, "{:.0f}", "labeling every scan, $ millions")
v("hosp_budget", m(SCANS.budget, "USD"), "{:,.0f}", "budget, $")
v("hosp_random", m(SCANS.random_labels, "count"), "{:,.0f}", "random labels the budget buys")
v("hosp_active", m(SCANS.active_labels, "count"), "{:,.0f}", "labels active learning needs")
v("hosp_mult", SCANS.label_ratio, "{:.0f}", "random / active labels")

print("== Creating data (Literature.DataSelectionResults, Scenarios.FixMatchCifar10Cost, FoundationModelTenTasks)")
v("autoaug_gpuh", lit(DSR.AutoAugmentImageNetSearchGpuHours), "{:,.0f}", "AutoAugment search, GPU-hours")
v("cutmix_pts", lit(DSR.CutMixResNet50Top1GainPts), "{:.2f}", "CutMix top-1 gain, points")
for labels, entry in (("40", DSR.FixMatchCifar10Error40LabelsPct), ("250", DSR.FixMatchCifar10Error250LabelsPct),
                      ("4k", DSR.FixMatchCifar10Error4000LabelsPct)):
    v(f"fm_acc_{labels}", 100 - lit(entry), "{:.1f}", f"FixMatch accuracy, {labels} labels, %")
cifar_train = m(CIFAR10.training_examples, "count")
v("cifar_train", cifar_train, "{:,.0f}", "CIFAR-10 training images")
v("fm_labels_4k", SSL.supervised_labels, "{:,.0f}", "labels, supervised baseline")
v("fm_labels_250", SSL.ssl_labels, "{:.0f}", "labels, FixMatch")
v("fm_share_pct", pct(SSL.supervised_labels / cifar_train), "{:.0f}", "4,000 of CIFAR-10, %")
v("fm_labels_40", lit(DSR.FixMatchCifar10LabelsSmall), "{:.0f}", "labels, FixMatch's smallest budget")
v("fm_per_class_word", lit(DSR.FixMatchCifar10LabelsSmall) / CIFAR10.num_classes, "words", "the same, per class, in prose")
v("ssl_sup_usd", m(SSL.supervised_total, "USD"), "{:,.0f}", "supervised labels plus compute, $")
v("ssl_usd", m(SSL.ssl_total, "USD"), "{:.0f}", "FixMatch labels plus compute, $")
v("ssl_compute_mult", SSL.compute_multiple, "{:.0f}", "FixMatch compute / supervised")
v("pretrain_gpuh", m(FOUNDATION.pretrain_compute, "hour"), "{:,.0f}", "pretraining, GPU-hours")
v("ft_gpuh", m(FOUNDATION.finetune_compute, "hour"), "{:.0f}", "fine-tune per task, GPU-hours")
v("scratch_gpuh", m(FOUNDATION.scratch_compute, "hour"), "{:,.0f}", "from scratch per task, GPU-hours")
v("crossover_tasks", math.ceil(FOUNDATION.crossover_tasks), "{:.0f}", "tasks after which pretraining pays")

print("== The machine (Scenarios.ResNet50HalfCoresetEpochH100, ResNet50EchoingH100)")
v("keep50_pct", pct(EPOCH.keep_fraction), "{:.0f}", "coreset, %")
v("epoch_s", m(EPOCH.epoch_time, "second"), "{:.0f}", "full epoch, s")
v("epoch_kj", m(EPOCH.epoch_energy, "kJ"), "{:.0f}", "full epoch, kJ")
v("pruned_s", m(EPOCH.pruned_epoch_time, "second"), "{:.0f}", "pruned epoch, s")
v("pruned_kj", m(EPOCH.pruned_epoch_energy, "kJ"), "{:.0f}", "pruned epoch, kJ")
v("path_saved_h", m(EPOCH.run_saving_time, "hour"), "{:.1f}", "90 epochs saved, h")
v("path_saved_h_round", m(EPOCH.run_saving_time, "hour"), "{:.0f}", "the same, whole hours (as the chapter rounds)")
v("path_saved_mj", m(EPOCH.run_saving_energy, "MJ"), "{:.0f}", "90 epochs saved, MJ")
v("path_ratio", EPOCH.saving_over_score_energy, "{:.0f}", "saved / scoring energy")
v("archive_gb", m(EPOCH.archive_bytes, "GB"), "{:.0f}", "ImageNet archive, GB")
v("nvme_gbs", m(EPOCH.hardware.storage.bandwidth, "GB/s"), "{:.0f}", "local NVMe, GB/s")
v("read_archive_s", m(EPOCH.storage_time, "second"), "{:.0f}", "archive read, s")
v("half_gb", m(EPOCH.materialized_bytes, "GB"), "{:.0f}", "materialized coreset, GB")
v("link_gb", m(EPOCH.link_bytes, "GB"), "{:.0f}", "decoded tensors over PCIe per epoch, GB")
v("link_s", m(EPOCH.link_time, "second"), "{:.0f}", "PCIe copy per epoch, s")
v("demand_ips", m(ECHO.accelerator_rate, "count/second"), "{:,.0f}", "H100 demand, images/s")
v("host_ips", m(ECHO.host_rate, "count/second"), "{:,.0f}", "host supply, images/s")
v("host_epoch_s", m(EPOCH.host_epoch_time, "second"), "{:.0f}", "epoch at the host rate, s")
v("tf_input_pct", lit(DSR.TfProfilerInputBoundPct), "{:.1f}", "profile: step time waiting for input, %")
v("tf_input_round", lit(DSR.TfProfilerInputBoundPct), "{:.0f}", "the same, whole percent")
v("tf_step_s", lit(DSR.TfProfilerStepSeconds), "{:.1f}", "profile: step time, s")
v("tf_compute_s", lit(DSR.TfProfilerComputeSeconds), "{:.1f}", "profile: device compute per step, s")
v("echo_idle_pct", pct(ECHO.idle_fraction), "{:.0f}", "accelerator waiting on input, %")
v("echo_busy_pct", pct(ECHO.utilization), "{:.0f}", "accelerator busy without echo, %")
v("echo_r", ECHO.ratio, "{:.2f}", "pipeline ratio r")
v("echo_e", ECHO.echo_factor, "{:.0f}", "echo factor")
v("echo_h_no", m(ECHO.run_time, "hour"), "{:.1f}", "90 epochs without echo, h")
v("echo_h_e", m(ECHO.echoed_run_time, "hour"), "{:.1f}", "90 epochs with echo, h")
v("choi_speedup", lit(DSR.ChoiSpeedupMax), "{:.2f}", "Choi et al., less wall time, times")
v("choi_acc", lit(DSR.ChoiTargetAccuracyPct), "{:.0f}", "Choi et al., target accuracy, %")
v("choi_ratio_word", lit(DSR.ChoiUpstreamRatio), "words", "upstream slowdown, in prose")
v("choi_e_min", lit(DSR.ChoiEchoFactorMin), "{:.0f}", "smallest echo factor")
v("choi_e_max", lit(DSR.ChoiEchoFactorMax), "{:.0f}", "largest echo factor")
v("choi_runs_word", lit(DSR.ChoiRuns), "words", "runs each, in prose")

print("== The bill (Scenarios.ImageNetDataCostBreakdown, MinHashPipelineAmortization)")
v("cost_acq", m(BILL.acquisition, "USD"), "{:,.0f}", "acquisition, $")
v("cost_labels", m(BILL.label_cost, "USD"), "{:,.0f}", "labels, $")
v("cost_label_cents", pct(m(BILL.label_price, "USD/count")), "{:.0f}", "label, cents")
v("cost_label_usd", m(BILL.label_price, "USD/count"), "{:.2f}", "label, $")
v("cost_storage", m(BILL.storage_cost, "USD"), "{:.0f}", "storage, $")
v("storage_months", m(BILL.storage_duration, "month"), "{:.0f}", "retention, months")
v("storage_cents", pct(m(BILL.storage_rate, "USD/GB/month")), "{:.0f}", "storage, cents per GB-month")
v("cost_runs", BILL.runs, "{:.0f}", "training runs")
v("cost_run_h", m(BILL.run_time, "hour"), "{:.1f}", "one run, h")
v("cost_gpuh", m(BILL.accelerator_hours, "hour"), "{:.0f}", "campaign, GPU-hours")
v("gpu_rate", m(BILL.accelerator_rate, "USD/hour"), "{:.0f}", "GPU-hour, $")
v("cost_compute", m(BILL.compute_cost, "USD"), "{:.0f}", "compute, $")
v("cost_total", m(BILL.total, "USD"), "{:,.0f}", "total, $")
v("cost_compute_pct", pct(m(BILL.compute_share, "")), "{:.1f}", "compute share, %")
v("cost_data_pct", pct(m(BILL.data_share, "")), "{:.1f}", "data share, %")
assert pct(m(BILL.compute_share, "")) < 1, "the vote's answer is 'under 1%'"
v("roi_invest", m(PIPELINE.investment, "USD"), "{:,.0f}", "investment, $")
v("roi_build", m(PIPELINE.build_cost, "USD"), "{:,.0f}", "build, $")
v("roi_operate", m(PIPELINE.operating_cost, "USD"), "{:,.0f}", "operate, $")
v("roi_save", m(PIPELINE.saving_per_run, "USD"), "{:,.0f}", "saved per run, $")
for runs in PIPELINE.run_counts:
    v(f"roi_mult_{runs}", PIPELINE.savings_multiple(runs), "{:.1f}", f"savings / investment after {runs} runs")
    v(f"roi_runs_{runs}_word", runs, "words", "the run count, in prose")
v("roi_breakeven", PIPELINE.breakeven_runs, "{:.1f}", "break-even, runs")
v("roi_payback_word", math.ceil(PIPELINE.breakeven_runs), "words", "first whole run past break-even, in prose")

print("== Fallacies (Scenarios.TenfoldDataFourPoints, ExpensiveSelectionPitfall)")
v("data_scale", SCALING.scale, "{:.0f}", "data multiple")
v("data_scale_word", SCALING.scale, "words", "the same, in prose")
v("data_gain", SCALING.accuracy_gain, "{:.0f}", "accuracy gain, points")
v("pitfall_sel_h", m(PITFALL.selection_time, "hour"), "{:.0f}", "expensive selection, h")
v("pitfall_full_h", m(PITFALL.full_time, "hour"), "{:.0f}", "full training, h")


if __name__ == "__main__":
    write(__file__)
