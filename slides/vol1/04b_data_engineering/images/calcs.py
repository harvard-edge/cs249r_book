#!/usr/bin/env python3
"""Every derived number in Lecture 4b (Data Engineering, Part 2), computed from sourced inputs.

One block per slide, in deck order. Inputs carry their source as a comment;
each printed line is a value the slide or its note shows, under the label
used there. make_figs.py imports the values below so figures and slides
share one computation.

"Chapter" means the rendered Volume I chapter Data Engineering. Units are
decimal: 1 KB = 1,000 bytes, 1 MB = 10^6 bytes, 1 GB = 10^9 bytes.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

KB, MB, GB, TB, PB = 1e3, 1e6, 1e9, 1e12, 1e15


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


def phi(x):
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------------------
# Moving a petabyte (Napkin Math 4.7)
# ---------------------------------------------------------------------------
DATASET_B = 1 * PB            # 1 PB (Napkin Math 4.7)
LINK_GBPS = 100               # dedicated link, Gb/s (Napkin Math 4.7)
SLOW_LINK_GBPS = 10           # the 10 Gb/s case (S4.9.1)
EGRESS_USD_PER_GB = 0.02      # AWS transfer pricing, 2024 (Napkin Math 4.7)
TPU_HOURS_CHAPTER = 5000      # "approximately 5,000 TPUv4-hours" (Napkin Math 4.7)
RUN_H = 20                    # the training run in the chapter's self-check

link_GBs = LINK_GBPS / 8
xfer_s = DATASET_B / GB / link_GBs
xfer_h = xfer_s / 3600
slow_days = DATASET_B / GB / (SLOW_LINK_GBPS / 8) / 86400
egress_usd = DATASET_B / GB * EGRESS_USD_PER_GB

# ---------------------------------------------------------------------------
# Energy of movement (chapter table 6)
# ---------------------------------------------------------------------------
PJ_FP32_MUL, PJ_DRAM, PJ_NET = 3.7, 640, 40_000   # table 6, 32-bit operand

# ---------------------------------------------------------------------------
# Storage tiers: the KWS corpus (S4.10.2, table 8)
# ---------------------------------------------------------------------------
KWS_CORPUS_GB = 748.8         # KWS reference corpus (S4.10.2)
OBJ_USD_TB_MONTH = 23         # within table 8's $20-25/TB-month; reproduces the chapter's $17.2
NVME_USD_TB_MONTH = (100, 300)  # table 8
OBJ_MBS = 100                 # object store, per connection (table 8, low end)
NVME_GBS = 5                  # local NVMe, low end (table 8)
REQ_S, READS_PER_REQ = 10_000, 10   # recommendation service (S4.10.1)

# ---------------------------------------------------------------------------
# ResNet-50 on one A100 (S4.9.3, Napkin Math 4.8)
# ---------------------------------------------------------------------------
A100_TFLOPS = 312             # A100 peak (S4.9.3)
TRAIN_GFLOP = 24.6            # per training image (S4.9.3)
SUSTAINED = 3200              # images/s sustained (S4.9.3)
JPEG_B = 150 * KB             # per JPEG (S4.9.3)
SUPPLY_MBS = 250              # attached cloud block volume (S4.9.3)
IMG_HW, IMG_C, FP32_B = 224, 3, 4   # decoded tensor (S4.9.3)
SATA_MBS = 500                # SATA SSD (Napkin Math 4.8)
N_ACCEL = 8                   # eight A100s (S4.10.1)
CHAPTER_CEILING = 12682       # the chapter prints 12,682 img/s

# NVIDIA DeepLearningExamples, PyTorch/Classification/ConvNets/resnet50v1.5/
# README.md (commit 5843f4e, 2022-10-03), "Training performance: NVIDIA DGX
# A100 (8x A100 80GB)", pytorch-21.03 NGC container, mixed precision.
NV_1GPU = 2470                # img/s, 1 GPU
NV_8GPU = 16621               # img/s, 8 GPUs

ceiling = A100_TFLOPS * 1e12 / (TRAIN_GFLOP * 1e9)
need_MBs = SUSTAINED * JPEG_B / MB
supplied_frac = SUPPLY_MBS / need_MBs
decoded_B = IMG_HW * IMG_HW * IMG_C * FP32_B
decoded_need_GBs = SUSTAINED * decoded_B / GB
decoded_loss = 1 - SUPPLY_MBS * MB / (SUSTAINED * decoded_B)
sata_imgs = SATA_MBS * MB / JPEG_B
ceiling_need_GBs = CHAPTER_CEILING * JPEG_B / GB
eight_imgs = N_ACCEL * SUSTAINED
eight_GBs = eight_imgs * JPEG_B / GB
eight_decoded_GBs = eight_imgs * decoded_B / GB

# ---------------------------------------------------------------------------
# Loader workers (S4.10.4)
# ---------------------------------------------------------------------------
PER_WORKER = 375              # images/s per loader worker (S4.10.4)
crossing = SUSTAINED / PER_WORKER
workers_one = math.ceil(crossing)

# ---------------------------------------------------------------------------
# PyTorch Profiler With TensorBoard tutorial (pytorch/tutorials,
# intermediate_source/tensorboard_profiler_tutorial.py at commit 45a5b17,
# the last version before its removal on 2026-06-01; images
# _static/img/profiler_overview1.png and profiler_overview2.png).
# ResNet-18 on CIFAR-10 resized to 224, batch 32, one Tesla V100.
# Execution Summary, average step, microseconds, as the tool prints them.
# ---------------------------------------------------------------------------
PROF = {
    "default loader": {"step": 132_827, "DataLoader": 77_528, "CPU Exec": 15_605,
                       "Kernel": 31_284, "dl_pct": 58.37, "gpu_util": 23.54},
    "num_workers=4": {"step": 76_170, "DataLoader": 768, "CPU Exec": 26_477,
                      "Kernel": 30_855, "dl_pct": 1.01, "gpu_util": 40.46},
}

# ---------------------------------------------------------------------------
# Columnar layout and codecs (Napkin Math 4.9, S4.10.5)
# ---------------------------------------------------------------------------
COLS_USED, COLS_ALL = 20, 100   # Napkin Math 4.9
EPOCH_GB = 100                  # S4.10.5
GZIP_MBS, SNAPPY_MBS = 120, 500  # decompression rates (S4.10.5)
EPOCHS = 50                     # S4.10.5

# A real Parquet file: NYC TLC Yellow Taxi trip records, January 2024,
# https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
# (49,961,641 bytes, retrieved 2026-10-18). Values printed by
# pyarrow 25.0.0: ParquetFile(path).metadata.row_group(0).column(i), and
# pyarrow.compute.count_distinct on the column.
PQ_ROWS_RG0 = 1_048_576
PQ_ROWS_FILE = 2_964_624
PQ_COLS = 19
PQ = {  # column: (physical type bytes, compressed, uncompressed-after-encoding, distinct)
    "payment_type": (8, 143_379, 257_706, 4),
    "PULocationID": (4, 741_555, 1_051_681, 247),
    "trip_distance": (8, 1_441_439, 1_605_038, 3_774),
}
PQ_RG0_COMPRESSED = 17_611_010   # sum of total_compressed_size, all 19 columns
PQ_TIMESTAMPS = {"tpep_pickup_datetime": 4_478_580, "tpep_dropoff_datetime": 4_563_191}
PQ_PICKUP_DISTINCT = 581_764     # count_distinct, row group 0

# ---------------------------------------------------------------------------
# A real WebDataset shard (github.com/webdataset/webdataset README, v0.2.107):
# https://storage.googleapis.com/webdataset/testdata/publaynet-train-000000.tar
# Content-Length 303,779,840 bytes; `tar tf | wc -l` prints 1,970 members
# (a .json and a .png per sample). Retrieved 2026-10-18.
# ---------------------------------------------------------------------------
SHARD_B = 303_779_840
SHARD_MEMBERS = 1970
FILES_PER_SAMPLE = 2

# ---------------------------------------------------------------------------
# Drift statistics (Napkin Math 4.5, S4.12.1)
# ---------------------------------------------------------------------------
KS_C = 1.36                   # 5 percent level (Napkin Math 4.5)
N_SMALL, N_LARGE = 1000, 100_000
D_OBS = 0.08                  # observed (Napkin Math 4.5)
PSI_BASE = [0.20] * 5         # training quintiles
PSI_RECENT = [0.30, 0.25, 0.20, 0.15, 0.10]   # illustrative shares (slide)

# The deck's illustrative loudness curves: normal, sd 6 dB, training mean
# -30 dBFS, recent mean -31.2 dBFS (make_figs.py). A "harmless" shift for the
# Evidently slide: 0.3 dB, illustrative.
LOUD_SD = 6.0
SHIFT_CHAPTER_DB = 1.2
SHIFT_SMALL_DB = 0.3
# Evidently v0.7.23, src/evidently/legacy/calculations/stattests/:
# registry.py _get_default_stattest (K-S at <= 1,000 reference rows,
# normed Wasserstein above), ks_stattest.py default_threshold=0.05 (p-value),
# wasserstein_distance_norm.py default_threshold=0.1 (distance / reference sd).
EV_ROWS_SWITCH = 1000
EV_WD_THRESHOLD = 0.1


def ks_crit(n0, n1):
    return KS_C * math.sqrt((n0 + n1) / (n0 * n1))


def ks_d_shift(delta_sd):
    """Largest CDF gap between two equal-sd normals whose means differ by delta_sd sds."""
    return 2 * phi(delta_sd / 2) - 1


psi_terms = [(r - b) * math.log(r / b) for r, b in zip(PSI_RECENT, PSI_BASE)]

# ---------------------------------------------------------------------------
# Speech Commands: two fixed releases. Warden, "Speech Commands: A Dataset
# for Limited-Vocabulary Speech Recognition," arXiv:1804.03209 (2018):
# S6 (v2: 105,829 utterances, 35 words, 2,618 speakers), S7.3 (v1, released
# 2017-08-03: 64,727 utterances, 1,881 speakers) and the Top-One table
# (Figure 2): the same default convolution model trained on each release
# and evaluated on each release's test set.
# ---------------------------------------------------------------------------
SC_V1_UTT, SC_V2_UTT = 64_727, 105_829
SC_V1_SPK, SC_V2_SPK = 1_881, 2_618
SC_TOP1 = {("V1 train", "V1 test"): 85.4, ("V2 train", "V1 test"): 89.7,
           ("V1 train", "V2 test"): 82.7, ("V2 train", "V2 test"): 88.2}


def main():
    head("Hook and goals")
    show("eight A100s, sustained img/s (goals card context)", f"{eight_imgs:,}")

    head("Moving a petabyte takes about a day and $20,000")
    show("link GB/s", f"{link_GBs:.1f}")
    show("transfer time s", f"{xfer_s:,.0f}")
    show("transfer time h", f"{xfer_h:.1f}  (slide: about 22 h)")
    show("egress USD", f"{egress_usd:,.0f}")
    show("10 Gb/s link, days", f"{slow_days:.2f}  (slide: 9.3 days)")
    show("implied TPU v4 price, USD/h (chapter's 5,000 h)", f"{egress_usd / TPU_HOURS_CHAPTER:.2f}")
    show("DRAM access / FP32 multiply energy", f"{PJ_DRAM / PJ_FP32_MUL:.0f}x")
    show("network transfer / FP32 multiply energy", f"{PJ_NET / PJ_FP32_MUL:,.0f}x")

    head("Storage got 10 orders of magnitude cheaper; the tier gaps stayed")
    root = str(ROOT / "mlsysim")
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from mlsysim import Literature
        rows = Literature.Series.StoragePricePerTB.entries
        first = next(r for r in rows if r["hdd"] is not None)
        last = rows[-1]
        show(f"hard disk {first['year']:.0f}, USD/TB", f"{first['hdd'] / 1e9:.0f}B")
        show(f"hard disk {last['year']:.0f}, USD/TB", f"{last['hdd']:.1f}")
        show("orders of magnitude (log10 ratio)", f"{math.log10(first['hdd'] / last['hdd']):.1f}")
        show("2023 DRAM / SSD", f"{last['dram'] / last['ssd']:.0f}x")
        show("2023 SSD / hard disk", f"{last['ssd'] / last['hdd']:.1f}x")
    except ImportError as e:  # mlsysim not on the path
        show("price series", f"unavailable ({e})")

    head("Serving needs low latency; training needs high bandwidth")
    show("random reads per second", f"{REQ_S * READS_PER_REQ:,}")
    show("KWS corpus, object store USD/month", f"{KWS_CORPUS_GB / 1000 * OBJ_USD_TB_MONTH:.1f}")
    lo, hi = (KWS_CORPUS_GB / 1000 * p for p in NVME_USD_TB_MONTH)
    show("KWS corpus, NVMe USD/month", f"{lo:.1f} to {hi:.1f}")
    show("KWS load time, object store s", f"{KWS_CORPUS_GB * GB / (OBJ_MBS * MB):,.0f}")
    show("KWS load time, NVMe s", f"{KWS_CORPUS_GB / NVME_GBS:.1f}")
    show("NVMe / object store speedup", f"{NVME_GBS * 1000 / OBJ_MBS:.0f}x")

    head("The A100 can consume 12,682 images a second; it sustains 3,200")
    show("arithmetic ceiling img/s", f"{ceiling:,.1f}  (rounds to {round(ceiling):,}; chapter prints {CHAPTER_CEILING:,})")
    show("sustained / ceiling", f"{SUSTAINED / CHAPTER_CEILING:.1%}")
    show("NVIDIA measured, 1 A100 / ceiling", f"{NV_1GPU / CHAPTER_CEILING:.1%}")
    show("NVIDIA measured / chapter's sustained", f"{NV_1GPU / SUSTAINED:.0%}")
    show("NVIDIA measured, 8 A100s, per GPU img/s", f"{NV_8GPU / 8:,.0f}")

    head("Your turn: can 250 MB/s keep ResNet-50 fed?")
    show("needed MB/s", f"{need_MBs:.0f}")
    show("supplied / needed", f"{supplied_frac:.2f}")
    show("training rate lost", f"{1 - supplied_frac:.0%}")

    head("Caching decoded tensors makes the loss worse")
    show("decoded bytes per image", f"{decoded_B:,}  ({decoded_B / KB:.0f} KB)")
    show("decoded / JPEG", f"{decoded_B / JPEG_B:.2f}x")
    show("decoded demand GB/s", f"{decoded_need_GBs:.2f}")
    show("decoded modeled loss", f"{decoded_loss:.0%}")

    head("A SATA SSD feeds today's rate, not the ceiling or eight A100s")
    show("SATA img/s", f"{sata_imgs:,.0f}")
    show("ceiling demand GB/s", f"{ceiling_need_GBs:.2f}")
    show("object-store streams at the ceiling", f"{ceiling_need_GBs * 1000 / OBJ_MBS:.1f}")
    show("SATA / ceiling", f"{sata_imgs / CHAPTER_CEILING:.1%}")
    show("eight A100s img/s", f"{eight_imgs:,}")
    show("eight A100s, compressed GB/s", f"{eight_GBs:.2f}")
    show("eight A100s, decoded GB/s", f"{eight_decoded_GBs:.2f}")

    head("With storage fixed, host decode binds next")
    show("crossing, workers", f"{crossing:.2f}  (whole workers: {workers_one})")
    show("eight A100s, one loader per accelerator, workers", f"{N_ACCEL * workers_one}")
    show("eight A100s, pooled workers", f"{math.ceil(eight_imgs / PER_WORKER)}")

    head("The profiler names the stall (PyTorch tutorial)")
    a, b = PROF["default loader"], PROF["num_workers=4"]
    for k, v in PROF.items():
        show(f"{k}: step ms", f"{v['step'] / 1000:.1f}")
        show(f"{k}: DataLoader ms", f"{v['DataLoader'] / 1000:.1f}")
        show(f"{k}: rest of step ms", f"{(v['step'] - v['DataLoader']) / 1000:.1f}")
        show(f"{k}: DataLoader share (recomputed)", f"{v['DataLoader'] / v['step']:.2%}  (tool: {v['dl_pct']}%)")
    show("step speedup", f"{a['step'] / b['step']:.2f}x")
    show("step time saved ms", f"{(a['step'] - b['step']) / 1000:.1f}")
    show("DataLoader time removed ms", f"{(a['DataLoader'] - b['DataLoader']) / 1000:.1f}")
    show("CPU Exec growth ms", f"{(b['CPU Exec'] - a['CPU Exec']) / 1000:.1f}")

    head("Columnar files read one fifth of the bytes")
    show("share read", f"{COLS_USED / COLS_ALL:.0%}  (one fifth; {COLS_ALL // COLS_USED}x less)")

    head("A real Parquet file (NYC taxi, January 2024)")
    t, comp, enc, nd = PQ["payment_type"]
    raw = PQ_ROWS_RG0 * t
    show("payment_type raw bytes (rows x 8)", f"{raw:,}  ({raw / MB:.1f} MB)")
    show("payment_type encoded bytes", f"{enc:,}  ({enc / KB:.0f} KB)")
    show("payment_type stored bytes", f"{comp:,}  ({comp / KB:.0f} KB)")
    show("payment_type raw / encoded", f"{raw / enc:.1f}x")
    show("payment_type raw / stored", f"{raw / comp:.1f}x")
    show("payment_type stored bits per row", f"{comp * 8 / PQ_ROWS_RG0:.2f}")
    show("payment_type distinct values", nd)
    used = sum(v[1] for v in PQ.values())
    show("3 of 19 columns, stored bytes", f"{used:,}  ({used / MB:.1f} MB)")
    show("3 of 19 columns, share of row group", f"{used / PQ_RG0_COMPRESSED:.1%}")
    show("row group 0 stored MB", f"{PQ_RG0_COMPRESSED / MB:.1f}")
    for k, v in PQ_TIMESTAMPS.items():
        show(f"{k} stored MB", f"{v / MB:.2f}")
    show("largest stored column", max(PQ_TIMESTAMPS, key=PQ_TIMESTAMPS.get))

    head("The right codec depends on which stage binds")
    gz = EPOCH_GB * GB / (GZIP_MBS * MB)
    sn = EPOCH_GB * GB / (SNAPPY_MBS * MB)
    show("gzip min per epoch", f"{gz / 60:.1f}")
    show("Snappy min per epoch", f"{sn / 60:.1f}")
    show("difference min per epoch", f"{(gz - sn) / 60:.1f}")
    show("50 epochs, hours of host CPU", f"{EPOCHS * (gz - sn) / 3600:.1f}  (chapter rounds to 9)")
    show("Snappy / gzip decompression rate", f"{SNAPPY_MBS / GZIP_MBS:.1f}x")

    head("Shards turn seeks into streams")
    samples = SHARD_MEMBERS // FILES_PER_SAMPLE
    show("samples in the shard", f"{samples:,}")
    show("shard MB", f"{SHARD_B / MB:.0f}")
    show("mean bytes per sample KB", f"{SHARD_B / samples / KB:.0f}")
    show("SATA img/s against sustained", f"{sata_imgs:,.0f} vs {SUSTAINED:,}")

    head("The K-S test flags drift when the CDFs separate")
    show("critical value, n = 1,000", f"{ks_crit(N_SMALL, N_SMALL):.4f}")
    show("D for the deck's 1.2 dB curve shift", f"{ks_d_shift(SHIFT_CHAPTER_DB / LOUD_SD):.4f}")

    head("At 100,000 clips, harmless shifts flag too")
    show("critical value, n = 100,000", f"{ks_crit(N_LARGE, N_LARGE):.4f}")

    head("PSI adds every bin that moved")
    for j, term in enumerate(psi_terms):
        show(f"bin {j + 1} term", f"{term:.4f}")
    show("PSI", f"{sum(psi_terms):.3f}")

    head("Evidently switches tests at 1,000 rows")
    for db in (SHIFT_SMALL_DB, SHIFT_CHAPTER_DB):
        d = ks_d_shift(db / LOUD_SD)
        wd = db / LOUD_SD   # a pure location shift moves every quantile by db
        show(f"{db} dB shift: K-S D", f"{d:.3f}")
        show(f"{db} dB shift: flags K-S at n = 1,000?", d > ks_crit(N_SMALL, N_SMALL))
        show(f"{db} dB shift: flags K-S at n = 100,000?", d > ks_crit(N_LARGE, N_LARGE))
        show(f"{db} dB shift: normed Wasserstein", f"{wd:.2f}")
        show(f"{db} dB shift: flags Wasserstein (>= {EV_WD_THRESHOLD})?", wd >= EV_WD_THRESHOLD)
    try:  # the same scipy calls Evidently makes, on simulated loudness
        import numpy as np
        from scipy import stats
        rng = np.random.default_rng(0)
        ref = rng.normal(-30, LOUD_SD, N_LARGE)
        cur = rng.normal(-30 - SHIFT_SMALL_DB, LOUD_SD, N_LARGE)
        ks = stats.ks_2samp(ref, cur)
        wd = stats.wasserstein_distance(ref, cur) / np.std(ref)
        show("simulated 0.3 dB, n = 100,000: scipy ks_2samp D, p", f"{ks.statistic:.3f}, {ks.pvalue:.1e}")
        show("simulated 0.3 dB, n = 100,000: normed Wasserstein", f"{wd:.3f}")
    except ImportError as e:
        show("scipy check", f"unavailable ({e})")

    head("Speech Commands: two fixed releases")
    s = SC_TOP1
    show("V1 test: V2 training minus V1 training", f"{s[('V2 train', 'V1 test')] - s[('V1 train', 'V1 test')]:+.1f} points")
    show("V2 test: V2 training minus V1 training", f"{s[('V2 train', 'V2 test')] - s[('V1 train', 'V2 test')]:+.1f} points")
    show("V1 training: V2 test minus V1 test", f"{s[('V1 train', 'V2 test')] - s[('V1 train', 'V1 test')]:+.1f} points")
    show("V2 training: V2 test minus V1 test", f"{s[('V2 train', 'V2 test')] - s[('V2 train', 'V1 test')]:+.1f} points")
    show("diagonal: (V2 train, V2 test) minus (V1 train, V1 test)", f"{s[('V2 train', 'V2 test')] - s[('V1 train', 'V1 test')]:+.1f} points")
    show("utterances v2 / v1", f"{SC_V2_UTT / SC_V1_UTT:.2f}x")
    show("speakers added", f"{SC_V2_SPK - SC_V1_SPK:,}")


if __name__ == "__main__":
    main()
