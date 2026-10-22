#!/usr/bin/env python3
"""Every number on the Lecture 4b (Data Engineering, Part 2) slides and notes, read from MLSysIM.

MLSysIM holds the inputs (Hardware, Models, Systems.Storage, Infrastructure
pricing, Ops.Monitoring, Literature) and the calculations (the delivery and
drift Scenarios.*); the chapter reads the same scenarios. This file only
selects values, converts units, and formats them; it types no numbers. Each
value is recorded with v() under the key the deck prints as \\val{key};
running this file writes images/numbers.tex. make_figs.py imports this module,
so figures and slides share one computation.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402

Hardware, Systems, Ops = mlsysim.Hardware, mlsysim.Systems, mlsysim.Ops
Literature, Scenarios = mlsysim.Literature, mlsysim.Scenarios
STORE, MON = Systems.Storage, Ops.Monitoring
PIPE, FILES = Literature.DataPipelineResults, Literature.DataFileMeasurements
SC, BENCH = Literature.SpeechCommandsReleases, Literature.Benchmarks

TARGETS, REGRESS = Scenarios.KeywordSpotterTargets, Scenarios.KeywordSpotterSilentRegression
MOVE = Scenarios.PetabyteCrossRegionMove
FEED, SERVER = Scenarios.ResNet50A100InputFeed, Scenarios.ResNet50EightA100InputFeed
LOOKUPS, TIERS = Scenarios.RecommendationFeatureLookups, Scenarios.MswcCorpusStorageTiers
COLS, CODECS = Scenarios.FraudModelColumnProjection, Scenarios.GzipVersusSnappyEpochs
KS, KS_BIG = Scenarios.LoudnessDriftKs1000, Scenarios.LoudnessDriftKs100k
SHIFT, SMALL = Scenarios.LoudnessShiftMatchingObservedGap, Scenarios.LoudnessShiftHarmless
BINS, KAPPA = Scenarios.QuieterMicrophoneLoudnessBins, Scenarios.WakeWordLabelQualityDrift
RECHT = Scenarios.ImageNetV2Shift

# The Parquet columns a model reads on the projection step, in slide order:
# (stored-bytes entry, slide name)
PQ_READ = [("TaxiTripDistanceStoredBytes", "trip distance"), ("TaxiPickupZoneStoredBytes", "pickup zone"),
           ("TaxiPaymentTypeStoredBytes", "payment type")]


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


def pct(x):
    """A fraction as percent."""
    return 100 * x


def lit(entry):
    """A Literature value as a plain number."""
    return float(entry)


def yes(flag):
    return "yes" if flag else "no"


print("== Hook and goals (Mohan et al. 2021; keyword-spotter scenarios)")
v("mohan_models", lit(PIPE.MohanModelsProfiled), "words", "vision and audio models profiled, in prose")
v("mohan_speedup", lit(PIPE.MohanCoordlSpeedupMax), "{:.0f}", "largest training-time cut, times")
v("mohan_speedup_word", lit(PIPE.MohanCoordlSpeedupMax), "words", "the same, in prose")
v("kws_detect_pct", pct(TARGETS.detection_target), "{:.0f}", "detection at launch, %")
v("regress_pts", pct(REGRESS.drop), "{:.0f}", "detection fall, points")
v("regress_pts_word", pct(REGRESS.drop), "words", "the same, in prose")
v("regress_months_word", m(REGRESS.elapsed, "month"), "words", "months after launch, in prose")
v("img_per_s", m(FEED.sustained_rate, "count/second"), "{:,.0f}", "ResNet-50 sustained images a second on one A100")
v("jpeg_kb", m(FEED.encoded_sample, "KB"), "{:.0f}", "JPEG as stored, KB")

print("== Moving a petabyte (cross-region transfer scenario)")
v("pb", m(MOVE.payload, "PB"), "{:.0f}", "dataset, PB")
v("pb_gb_exp", round(math.log10(m(MOVE.payload, "GB"))), "{:.0f}", "dataset, power of ten GB")
bits_per_byte = m(mlsysim.core.units.Q_(1, "byte"), "bit")
v("bits_per_byte", bits_per_byte, "{:.0f}", "bits per byte")
v("link_gbps", m(MOVE.links["100 Gb/s"], "Gbps"), "{:.0f}", "dedicated link, Gb/s")
v("link_gbs", m(MOVE.links["100 Gb/s"], "GB/s"), "{:.1f}", "dedicated link, GB/s")
v("xfer_s", m(MOVE.transfer_time("100 Gb/s"), "second"), "{:,.0f}", "transfer time, s")
v("xfer_h", m(MOVE.transfer_time("100 Gb/s"), "hour"), "{:.0f}", "transfer time, h")
v("egress_rate", m(MOVE.egress_rate, "USD/GB"), "{:.2f}", "egress, $ per GB")
v("egress_cents_word", pct(m(MOVE.egress_rate, "USD/GB")), "words", "egress, cents per GB, in prose")
v("egress_usd", m(MOVE.egress_cost, "USD"), "{:,.0f}", "egress for the move, $")
v("tpu_hours", m(MOVE.egress_compute_hours, "hour"), "{:,.0f}", "TPU v4 hours the egress buys")
v("slow_gbps", m(MOVE.links["10 Gb/s"], "Gbps"), "{:.0f}", "slower link, Gb/s")
v("slow_days", m(MOVE.transfer_time("10 Gb/s"), "day"), "{:.1f}", "transfer time on the slower link, days")
v("run_h", m(MOVE.job_time, "hour"), "{:.0f}", "training run, h")
dram, flop32 = Hardware.Tech.Memory.DRAM.energy_per_access, Hardware.Tech.Op.FlopFp32.energy
net = Hardware.Tech.Movement.Network.energy_per_byte
v("dram_mult", m(dram, "pJ") / m(flop32, "pJ/flop"), "{:.0f}", "DRAM access / FP32 multiply energy")
v("net_mult", m(net, "pJ/bit") / (m(flop32, "pJ/flop") / m(mlsysim.core.units.BYTES_FP32, "bit")),
  "{:,.0f}", "network transfer / FP32 multiply energy, per bit")

print("== Storage price per TB (vol1/04_data_engineering/data/storage_price_per_tb.csv)")
rows = load_series("vol1/04_data_engineering/data/storage_price_per_tb.csv").entries
first = next(r for r in rows if r["hdd"] is not None)
last = rows[-1]
v("hdd_first_year", first["year"], "{:.0f}", "first hard-disk year")
v("hdd_first_b", first["hdd"] / 10 ** 9, "{:.0f}", "hard disk then, billions of $ per TB")
v("hdd_last_year", last["year"], "{:.0f}", "latest year")
v("hdd_last", last["hdd"], "{:.1f}", "hard disk now, $ per TB")
v("hdd_orders", math.log10(first["hdd"] / last["hdd"]), "{:.0f}", "orders of magnitude")
v("hdd_orders_word", round(math.log10(first["hdd"] / last["hdd"])), "words", "the same, in prose")
v("dram_over_ssd", last["dram"] / last["ssd"], "{:.0f}", "DRAM / SSD, latest year")
v("ssd_over_hdd", last["ssd"] / last["hdd"], "{:.1f}", "SSD / hard disk, latest year")

print("== Serving against training (feature-lookup and corpus-tier scenarios; Systems.Storage)")
v("serve_rps", m(LOOKUPS.request_rate, "1/second"), "{:,.0f}", "requests a second")
v("reads_per_req", LOOKUPS.reads_per_request, "{:.0f}", "feature reads per request")
v("reads_per_req_word", LOOKUPS.reads_per_request, "words", "the same, in prose")
v("serve_iops", LOOKUPS.reads_per_second, "{:,.0f}", "random reads a second")
v("nvme_lat_us", m(STORE.LocalNvmeGen4.latency, "microsecond"), "{:.0f}", "local NVMe access latency, us")
v("dram_lat_ns", m(Hardware.Tech.Memory.HostDram_Latency2009.latency, "ns"), "{:.0f}", "host DRAM access latency, ns")
v("obj_mbs", m(STORE.ObjectStoreSingleStream.bandwidth, "MB/s"), "{:.0f}", "object store, one connection, MB/s")
v("nvme_lo", m(STORE.LocalNvmeTrainingTypical.bandwidth, "GB/s"), "{:.0f}", "local NVMe, low, GB/s")
v("nvme_hi", m(STORE.LocalNvmeGen4.bandwidth, "GB/s"), "{:.0f}", "local NVMe, high, GB/s")
v("kws_gb", m(TIERS.corpus_bytes, "GB"), "{:.1f}", "KWS corpus, GB")
v("kws_obj_usd", m(TIERS.monthly_cost("object store"), "USD/month"), "{:.1f}", "object store, $ a month")
v("kws_nvme_usd_lo", m(TIERS.monthly_cost("NVMe low"), "USD/month"), "{:.1f}", "NVMe, low, $ a month")
v("kws_nvme_usd_hi", m(TIERS.monthly_cost("NVMe high"), "USD/month"), "{:.1f}", "NVMe, high, $ a month")
v("kws_obj_s", m(TIERS.load_time("object store"), "second"), "{:,.0f}", "load from object store, s")
v("kws_nvme_s", m(TIERS.load_time("NVMe"), "second"), "{:.1f}", "load from NVMe, s")

print("== The accelerator's appetite (ResNet-50 A100 feed scenario; NVIDIA measurement)")
ceiling = int(m(FEED.ceiling_rate, "count/second"))   # whole images, as the chapter prints it
v("ceiling_img", ceiling, "{:,.0f}", "arithmetic ceiling, images a second")
v("peak_tflops", m(FEED.hardware.compute.peak_flops, "TFLOP/s"), "{:.0f}", "A100 peak, TFLOP/s")
v("train_gflop", m(FEED.training_flops_per_sample, "GFLOP"), "{:.1f}", "training FLOPs per image, GFLOP")
v("train_factor_word", FEED.training_flops_factor, "words", "training / prediction FLOPs, in prose")
v("sustained_pct", pct(round(FEED.sustained_fraction, 2)), "{:.0f}", "sustained / ceiling, % (as the chapter rounds)")
nv1 = lit(BENCH.ResNet50A100PyTorchOneGpuThroughput)
v("nv_img", nv1, "{:,.0f}", "NVIDIA measured, one A100, images a second")
v("nv_pct_ceiling", pct(nv1 / ceiling), "{:.0f}", "measured / ceiling, %")
v("nv_pct_sustained", pct(nv1 / m(FEED.sustained_rate, "count/second")), "{:.0f}", "measured / sustained, %")

print("== Can 250 MB/s keep ResNet-50 fed? (feed scenario)")
v("supply_mbs", m(FEED.supply.bandwidth, "MB/s"), "{:.0f}", "attached block volume, MB/s")
v("need_mbs", m(FEED.demand(), "MB/s"), "{:.0f}", "needed at the sustained rate, MB/s")
v("supplied_frac", FEED.supplied_fraction, "{:.2f}", "supplied / needed")
v("supplied_pct", pct(FEED.supplied_fraction), "{:.0f}", "the same, %")
v("loss_pct", pct(FEED.feeding_loss()), "{:.0f}", "training rate lost, %")

print("== Caching decoded tensors (feed scenario)")
v("decoded_kb", m(FEED.decoded_sample, "KB"), "{:.0f}", "decoded FP32 image, KB")
v("decode_mult", FEED.decode_expansion, "{:.0f}", "decoded / JPEG")
v("decode_mult_word", round(FEED.decode_expansion), "words", "the same, in prose")
v("decoded_gbs", m(FEED.demand(decoded=True), "GB/s"), "{:.1f}", "decoded demand, GB/s")
v("tensor_loss_pct", pct(FEED.feeding_loss(decoded=True)), "{:.0f}", "decoded modeled loss, %")

print("== Storage supply against demand (feed scenarios; Systems.Storage)")
sata = STORE.SataSsdEffective.bandwidth
sata_img = int(m(FEED.samples_supplied(sata), "count/second"))   # whole images, as the chapter prints it
v("sata_mbs", m(sata, "MB/s"), "{:.0f}", "SATA SSD, MB/s")
v("sata_img", sata_img, "{:,.0f}", "SATA SSD, images a second")
v("sata_pct", round(pct(sata_img / ceiling)), "{:.0f}", "SATA / ceiling, %")
v("ceiling_gbs", m(FEED.demand(rate=ceiling * FEED.sustained_rate.units), "GB/s"), "{:.1f}",
  "needed at the ceiling, GB/s")
v("obj_streams", round(FEED.streams_needed(STORE.ObjectStoreSingleStream.bandwidth,
                                           rate=ceiling * FEED.sustained_rate.units)),
  "{:.0f}", "object-store streams at the ceiling")
v("eight_img", m(SERVER.total_rate, "count/second"), "{:,.0f}", "eight A100s, images a second")
v("eight_gbs", m(SERVER.demand(), "GB/s"), "{:.1f}", "eight A100s, compressed, GB/s")
v("eight_decoded_gbs", m(SERVER.demand(decoded=True), "GB/s"), "{:.1f}", "eight A100s, decoded, GB/s")

print("== Host decode (feed scenario: loader workers)")
v("worker_rate", m(FEED.worker_rate, "count/second"), "{:.0f}", "images a second per loader worker")
v("workers_word", FEED.workers_per_accelerator, "words", "workers per accelerator, in prose")
v("eight_workers", SERVER.accelerators * FEED.workers_per_accelerator, "{:.0f}",
  "eight A100s, nine whole workers each")
v("host_cap", m(FEED.host_decode_cap, "count/second"), "{:,.0f}", "illustrative host decode cap, images a second")

print("== The profiler names the stall (PyTorch tutorial, Literature.DataPipelineResults)")
step_a, step_b = lit(PIPE.ProfilerDefaultStepUs), lit(PIPE.ProfilerFourWorkersStepUs)
dl_a, dl_b = lit(PIPE.ProfilerDefaultDataLoaderUs), lit(PIPE.ProfilerFourWorkersDataLoaderUs)
us_per_ms = 10 ** 3
v("prof_workers", lit(PIPE.ProfilerWorkers), "{:.0f}", "loader workers, second run")
v("prof_workers_word", lit(PIPE.ProfilerWorkers), "words", "the same, in prose")
v("prof_dl_pct_round", lit(PIPE.ProfilerDefaultDataLoaderPct), "{:.0f}", "DataLoader share, default, whole %")
v("prof_dl_pct", lit(PIPE.ProfilerDefaultDataLoaderPct), "{:.1f}", "DataLoader share, default, % (tool)")
v("prof_dl_pct_b", lit(PIPE.ProfilerFourWorkersDataLoaderPct), "{:.0f}", "DataLoader share, four workers, %")
v("prof_step_us", step_a, "{:,.0f}", "default step, us")
v("prof_dl_us", dl_a, "{:,.0f}", "default DataLoader, us")
v("prof_step_a", step_a / us_per_ms, "{:.1f}", "default step, ms")
v("prof_step_b", step_b / us_per_ms, "{:.1f}", "four-worker step, ms")
v("prof_speedup", step_a / step_b, "{:.2f}", "step speedup")
v("prof_dl_saved", (dl_a - dl_b) / us_per_ms, "{:.1f}", "DataLoader time removed, ms")
v("prof_step_saved", (step_a - step_b) / us_per_ms, "{:.1f}", "step time saved, ms")
v("prof_rest_a", (step_a - dl_a) / us_per_ms, "{:.1f}", "rest of step, default, ms")
v("prof_rest_b", (step_b - dl_b) / us_per_ms, "{:.1f}", "rest of step, four workers, ms")
v("prof_cpu_growth", (lit(PIPE.ProfilerFourWorkersCpuExecUs) - lit(PIPE.ProfilerDefaultCpuExecUs)) / us_per_ms,
  "{:.1f}", "CPU Exec growth, ms")
v("prof_gpu_a", lit(PIPE.ProfilerDefaultGpuUtilPct), "{:.1f}", "GPU utilization, default, %")
v("prof_gpu_b", lit(PIPE.ProfilerFourWorkersGpuUtilPct), "{:.1f}", "GPU utilization, four workers, %")

print("== Columnar projection (column-projection scenario)")
v("cols_used", COLS.columns_used, "{:.0f}", "columns the model uses")
v("cols_all", COLS.columns_total, "{:.0f}", "columns in the table")
v("col_speedup_word", COLS.speedup, "words", "effective-bandwidth gain, in prose")
v("col_speedup", COLS.speedup, "{:.0f}", "the same, times")

print("== A real Parquet file (NYC taxi, January 2024; Literature.DataFileMeasurements)")
rg_rows = lit(FILES.TaxiRowGroup0Rows)
raw = rg_rows * lit(FILES.TaxiPaymentTypeValueBytes)
enc, stored = lit(FILES.TaxiPaymentTypeEncodedBytes), lit(FILES.TaxiPaymentTypeStoredBytes)
used = sum(lit(getattr(FILES, k)) for k, _ in PQ_READ)
v("pq_cols", lit(FILES.TaxiParquetColumns), "{:.0f}", "columns in the file")
v("pq_rows_word", round(lit(FILES.TaxiParquetRows) / 10 ** 6), "words", "rows, millions, in prose")
v("pq_rows_raw", lit(FILES.TaxiParquetRows), "{:.0f}", "rows, as the tool prints")
v("pq_rg_rows_raw", rg_rows, "{:.0f}", "row-group rows, as the tool prints")
v("pq_rg_rows", rg_rows, "{:,.0f}", "row-group rows")
v("pq_enc_raw", enc, "{:.0f}", "payment_type after encoding, bytes, as the tool prints")
v("pq_stored_raw", stored, "{:.0f}", "payment_type stored, bytes, as the tool prints")
v("pq_value_bytes", lit(FILES.TaxiPaymentTypeValueBytes), "{:.0f}", "payment_type value width, bytes")
v("pq_distinct", lit(FILES.TaxiPaymentTypeDistinct), "{:.0f}", "payment_type distinct values")
v("pq_raw_mb", raw / 10 ** 6, "{:.1f}", "payment_type as raw INT64, MB")
v("pq_enc_kb", enc / 10 ** 3, "{:.0f}", "after dictionary and RLE, KB")
v("pq_stored_kb", stored / 10 ** 3, "{:.0f}", "after ZSTD, KB")
v("pq_enc_mult", raw / enc, "{:.1f}", "raw / encoded")
v("pq_stored_mult", raw / stored, "{:.1f}", "raw / stored")
v("pq_bits", stored * bits_per_byte / rg_rows, "{:.2f}", "stored bits per row")
v("pq_bits_round", stored * bits_per_byte / rg_rows, "{:.0f}", "the same, whole bits")
v("pq_cols_used", len(PQ_READ), "{:.0f}", "columns the model reads")
v("pq_used_mb", used / 10 ** 6, "{:.1f}", "their stored bytes, MB")
v("pq_rg_mb", lit(FILES.TaxiRowGroup0StoredBytes) / 10 ** 6, "{:.1f}", "row group stored, MB")
v("pq_used_pct", pct(used / lit(FILES.TaxiRowGroup0StoredBytes)), "{:.1f}", "share of the row group read, %")
v("pq_ts_distinct", lit(FILES.TaxiPickupTimeDistinct), "{:,.0f}", "distinct pickup times, row group 0")
ts = (lit(FILES.TaxiPickupTimeStoredBytes), lit(FILES.TaxiDropoffTimeStoredBytes))
v("pq_ts_mb", sum(ts) / len(ts) / 10 ** 6, "{:.1f}", "each timestamp column stored, MB (mean of the two)")

print("== Codecs (gzip-versus-Snappy scenario)")
v("codec_gb", m(CODECS.epoch_bytes, "GB"), "{:.0f}", "bytes per epoch, GB")
v("gzip_mbs", m(CODECS.decode_rates["gzip"], "MB/s"), "{:.0f}", "gzip decompression, MB/s")
v("snappy_mbs", m(CODECS.decode_rates["Snappy"], "MB/s"), "{:.0f}", "Snappy decompression, MB/s")
v("gzip_min", m(CODECS.decode_time("gzip"), "minute"), "{:.1f}", "gzip, min per epoch")
v("snappy_min", m(CODECS.decode_time("Snappy"), "minute"), "{:.1f}", "Snappy, min per epoch")
v("codec_epochs", CODECS.epochs, "{:.0f}", "epochs")
v("codec_h", m(CODECS.run_time_saved("gzip", "Snappy"), "hour"), "{:.1f}", "host CPU saved over the run, h")
v("codec_h_round", round(m(CODECS.run_time_saved("gzip", "Snappy"), "hour")), "{:.0f}",
  "the same, whole hours (as the chapter rounds)")

print("== Shards (WebDataset PubLayNet shard; Literature.DataFileMeasurements and DataPipelineResults)")
members = lit(FILES.PublaynetShardMembers)
v("shard_members_raw", members, "{:.0f}", "files in the shard, as the tool prints")
v("shard_members", members, "{:,.0f}", "files in the shard")
v("shard_samples", members / lit(FILES.PublaynetFilesPerSample), "{:,.0f}", "samples in the shard")
v("shard_mb", lit(FILES.PublaynetShardBytes) / 10 ** 6, "{:.0f}", "shard size, MB")
v("shuffle_buf", lit(FILES.WebDatasetShuffleBufferSamples), "{:,.0f}", "suggested shuffle buffer, samples")
v("ffcv_pct", lit(PIPE.FfcvResNet50Top1Pct), "{:.0f}", "FFCV ResNet-50 top-1 target, %")
v("ffcv_min", lit(PIPE.FfcvResNet50Minutes), "{:.0f}", "FFCV time on one machine, minutes")

print("== Drift (ImageNetV2 scenario; K-S, location-shift, and PSI scenarios; Ops.Monitoring)")
r = RECHT.result
v("recht_networks", r.model_count - r.baseline_count, "{:.0f}", "ImageNetV2 neural networks")
v("recht_net_gap_min", r.network_gap_min, "{:.1f}", "smallest network drop, points")
v("recht_net_gap_max", r.network_gap_max, "{:.1f}", "largest network drop, points")
v("ks_c", KS.coefficient, "{:.2f}", "K-S coefficient at the 0.05 level")
v("ks_n", KS.n_reference, "{:,.0f}", "clips per sample")
v("ks_nsum", KS.n_reference + KS.n_current, "{:.0f}", "n0 + nt")
v("ks_nprod_exp", round(math.log10(KS.n_reference * KS.n_current)), "{:.0f}", "n0 nt, power of ten")
v("ks_crit", KS.critical_value, "{:.3f}", "critical value, n = 1,000")
v("ks_d", KS.observed_statistic, "{:.2f}", "observed largest CDF gap")
v("ks_n_big", KS_BIG.n_reference, "{:,.0f}", "clips per sample, whole traffic")
v("ks_n_big_exp", round(math.log10(KS_BIG.n_reference)), "{:.0f}", "the same, power of ten")
v("ks_crit_big", KS_BIG.critical_value, "{:.3f}", "critical value, n = 100,000")
v("loud_sd", SHIFT.feature_sd, "{:.0f}", "loudness sd, dB")
for key, s in (("big", SHIFT), ("small", SMALL)):
    v(f"shift_{key}", abs(s.shift), "{:.1f}", f"{key} loudness shift, dB")
    v(f"shift_{key}_d", s.ks_statistic, "{:.3f}", f"{key} shift, K-S D")
    v(f"shift_{key}_flag_ks", yes(KS_BIG.flags_statistic(s.ks_statistic)), "{}", f"{key} shift flagged by K-S at n = 100,000")
    v(f"shift_{key}_wd", s.normed_wasserstein, "{:.2f}", f"{key} shift, normed Wasserstein")
    v(f"shift_{key}_flag_wd", yes(s.normed_wasserstein >= MON.EvidentlyWassersteinThreshold), "{}",
      f"{key} shift flagged by Wasserstein")
v("ev_rows_raw", MON.EvidentlyKsMaxReferenceRows, "{:.0f}", "Evidently K-S row limit, as the source prints")
v("ev_rows", MON.EvidentlyKsMaxReferenceRows, "{:,.0f}", "the same, with a comma")
v("ev_p", MON.EvidentlyKsPValueThreshold, "{:.2f}", "Evidently K-S p-value threshold")
v("ev_wd", MON.EvidentlyWassersteinThreshold, "{:.1f}", "Evidently normed Wasserstein threshold")
v("psi", BINS.psi, "{:.3f}", "PSI over five bins")
v("psi_base_pct", pct(BINS.baseline_shares[0]), "{:.0f}", "training share per bin, %")
for i, share in enumerate(BINS.current_shares):
    v(f"psi_recent_{i + 1}", pct(share), "{:.0f}", f"recent share, bin {i + 1}, %")
first_bin, last_bin = 0, len(BINS.current_shares) - 1
for key, i in (("quiet", first_bin), ("loud", last_bin)):
    v(f"psi_{key}_diff", abs(BINS.current_shares[i] - BINS.baseline_shares[i]), "{:.2f}", f"{key}est bin, share difference")
    v(f"psi_{key}_ratio", max(BINS.ratio(i), 1 / BINS.ratio(i)), "{:.1f}" if key == "quiet" else "{:.0f}",
      f"{key}est bin, share ratio inside the log")
    v(f"psi_{key}_term", BINS.terms[i], "{:.3f}", f"{key}est bin, PSI term")
v("psi_warn", MON.PsiWarnThreshold, "{:.1f}", "PSI warning threshold")
v("psi_crit", MON.PsiCriticalThreshold, "{:.2f}", "PSI critical threshold")
v("kappa_base", KAPPA.baseline_kappa, "{:.2f}", "campaign kappa, baseline")
v("kappa_later", KAPPA.later_kappa, "{:.2f}", "campaign kappa, later")
v("kappa_months_word", m(KAPPA.elapsed, "month"), "words", "months between, in prose")

print("== Speech Commands releases (Warden 2018, Literature.SpeechCommandsReleases)")
a, b = lit(SC.Top1V1TrainV1TestPct), lit(SC.Top1V2TrainV1TestPct)
c, d = lit(SC.Top1V1TrainV2TestPct), lit(SC.Top1V2TrainV2TestPct)
v("sc_v1v1", a, "{:.1f}", "trained v1, v1 test, %")
v("sc_v2v1", b, "{:.1f}", "trained v2, v1 test, %")
v("sc_v1v2", c, "{:.1f}", "trained v1, v2 test, %")
v("sc_v2v2", d, "{:.1f}", "trained v2, v2 test, %")
v("sc_data_gain", b - a, "{:.1f}", "v2 data on the v1 test set, points")
v("sc_test_harder", a - c, "{:.1f}", "v1 model, v2 test set lower by, points")
v("sc_diagonal", d - a, "{:.1f}", "diagonal difference, points")
v("sc_v1_utt", lit(SC.V1Utterances), "{:,.0f}", "v1 clips")
v("sc_v1_spk", lit(SC.V1Speakers), "{:,.0f}", "v1 speakers")
v("sc_v2_utt", lit(SC.V2Utterances), "{:,.0f}", "v2 clips")
v("sc_v2_words", lit(SC.V2Words), "{:.0f}", "v2 words")
v("sc_v2_spk", lit(SC.V2Speakers), "{:,.0f}", "v2 speakers")


if __name__ == "__main__":
    write(__file__)
