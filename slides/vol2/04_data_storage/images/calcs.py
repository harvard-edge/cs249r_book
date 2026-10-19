#!/usr/bin/env python3
"""Every derived number in Lecture 4 (Data Storage), computed from sourced inputs.

Run from this folder: python3 calcs.py
Each printed line is "slide label: value"; the deck and its notes copy these
values. Inputs cite the rendered chapter (Data Storage, Volume II, Chapter 4),
MLSysIM, or an outside source. make_figs.py imports this module for the values
it draws.
"""

import math
import statistics
import sys
from pathlib import Path

GB, TB, PB = 1e9, 1e12, 1e15

# --- Running case (Sec. 4.1) -------------------------------------------------
PARAMS = 175e9                 # 175B-parameter model (Sec. 4.1)
NODES, GPUS_PER_NODE = 256, 8  # running cluster (Sec. 4.1)
TOKENS = 1.5e12                # training tokens (Sec. 4.1)
BYTES_PER_TOKEN = 4            # 4-byte token IDs (Sec. 4.1)
DAYS = 30                      # run length (Sec. 4.1)
SAVE_INTERVAL_S = 600          # one checkpoint every 10 minutes (Sec. 4.1)
B_FP16, B_MASTER, B_ADAM = 2, 4, 8   # bytes per parameter (Sec. 4.1, 4.6)

GPUS = NODES * GPUS_PER_NODE
CORPUS = TOKENS * BYTES_PER_TOKEN
CKPT_BPP = B_FP16 + B_MASTER + B_ADAM
CKPT = PARAMS * CKPT_BPP
SAVES = DAYS * 24 * 3600 // SAVE_INTERVAL_S
CKPT_TOTAL = CKPT * SAVES
WRITE_READ_RATIO = CKPT_TOTAL / CORPUS
PER_NODE_CKPT = CKPT / NODES

# --- Tier ladder (Sec. 4.3.1 figure, table 2, Napkin Math 4.7) ---------------
HBM_BW, DRAM_BW, NVME4_BW, PFS_BW = 3350 * GB, 200 * GB, 28 * GB, 1000 * GB
PFS_SHARE = PFS_BW / NODES
R_HBM_DRAM = HBM_BW / DRAM_BW
R_DRAM_NVME = DRAM_BW / NVME4_BW
R_NVME_PFS = NVME4_BW / PFS_SHARE

# --- Cache (Sec. 4.3.1 eq. 1; Sec. 4.4.3) -----------------------------------
H1, MISS1 = 0.95, 100
ET1 = H1 * 1 + (1 - H1) * MISS1                # in units of T_hot
FRAC1 = 1 / ET1
MM_CORPUS, NODE_NVME = 35 * TB, 30.7 * TB      # Sec. 4.3.4 and 4.4.3
H2 = NODE_NVME / MM_CORPUS
T_PER_BYTE = H2 / NVME4_BW + (1 - H2) / (4 * GB)   # 4 GB/s per-node PFS share (Sec. 4.4.3)
RATE2 = 1 / T_PER_BYTE
FRAC2 = RATE2 / NVME4_BW
MISS2 = NVME4_BW / (4 * GB)

# --- I/O wall (Systems Perspective 4.1) --------------------------------------
DELIVERY_FRACTION = 0.25        # delivery takes a quarter of the original compute time
PLATEAU = 1 / DELIVERY_FRACTION
SPEEDUP_AT_16 = min(16, PLATEAU)

# --- Input demand (Napkin Math 4.1; Sec. 4.3.1) ------------------------------
STEP = 0.2
TEXT_BW = GPUS * 4096 * BYTES_PER_TOKEN / STEP
IMG_KB = 115.7e3               # bytes per ImageNet JPEG (Napkin Math 4.1)
IMG_BW = GPUS * 256 * IMG_KB / STEP
IMG_TEXT_RATIO = IMG_BW / TEXT_BW
JOB_AVG_BW = CORPUS / (DAYS * 24 * 3600)
VIDEO_CLIP = 1e6               # a 1 MB clip, "megabytes per sample" (Sec. 4.3.1)
VIDEO_OVER_IMAGE = VIDEO_CLIP / IMG_KB
IMG_SHARE_OF_PFS = IMG_BW / PFS_BW

# --- MLPerf Storage v1.0 (MLSysIM, MLCommons 2024) ----------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))
from mlsysim import Literature  # noqa: E402

ROWS = [r for r in Literature.Series.MLPerfStorageV1Training.entries
        if r["division"] == "Closed" and r["au_check"] == "pass"]
WORKLOADS = (("unet3d", "3D U-Net"), ("cosmoflow", "CosmoFlow"), ("resnet50", "ResNet-50"))
MED = {(w, a): statistics.median(r["io_GBps"] / r["num_accelerators"] for r in ROWS
                                 if r["workload"] == w and r["accelerator"] == a)
       for w, _ in WORKLOADS for a in ("A100", "H100")}
GEN = [MED[(w, "H100")] / MED[(w, "A100")] for w, _ in WORKLOADS]
EXAMPLE_ROW = next(r for r in ROWS if r["submitter"] == "DDN" and r["workload"] == "cosmoflow"
                   and r["accelerator"] == "H100" and r["num_accelerators"] == 64)

# --- Stall and prefetch (Napkin Math 4.5; Sec. 4.4.1) -------------------------
T_COMP, T_IO = 200, 250
SEQ_STEP = T_COMP + T_IO
SEQ_STALL = T_IO / SEQ_STEP
OVL_STEP = max(T_COMP, T_IO)
OVL_STALL = (OVL_STEP - T_COMP) / OVL_STEP
P99_OBJ, P99_LOCAL = 500, 50
PREFETCH_SPAN = P99_OBJ / T_COMP
PREFETCH_DEPTH = math.ceil(PREFETCH_SPAN)
PREFETCH_LOCAL = math.ceil(P99_LOCAL / T_COMP)
BATCH_MB, DEPTH5 = 40, 5        # decoded image batch about 40 MB; depth 5 (Sec. 4.4.1)
PREFETCH_NODE_GB = BATCH_MB * DEPTH5 * GPUS_PER_NODE / 1000

# --- GPUDirect Storage (Napkin Math 4.6, illustrative service times) ---------
IMG_RATE_NODE = 8000 * GPUS_PER_NODE
CPU_BUF = IMG_RATE_NODE * 120e-6
CPU_GDS = IMG_RATE_NODE * 30e-6
CORES_FREED = round(CPU_BUF) - round(CPU_GDS)

# --- Metadata and tails (Example 4.1; Sec. 4.3.6) ----------------------------
FILES, SHARDS = 200e6, 50_000
FILE_REDUCTION = FILES / SHARDS
OSS, OSS_BW = 100, 10 * GB
PFS_AGG = OSS * OSS_BW
P_ALL_FAST = 0.99 ** 100
P_SOME_SLOW = 1 - P_ALL_FAST
RUN_SHARDS = 2000              # about 2,000 shards of 3 GB (Sec. 4.3.5)

# --- 3FS (DeepSeek 3FS README, read stress test) ------------------------------
FS3_TIB_S, FS3_NODES = 6.6, 180
FS3_PER_NODE_GIB = FS3_TIB_S * 1024 / FS3_NODES
TIB_OVER_TB = 2 ** 40 / 1e12

# --- Checkpoint storm (Definition 4.5) and staging (Napkin Math 4.7) ----------
P70 = 70e9
STATE70 = P70 * CKPT_BPP
REPLICAS, INGEST = 1024, 100 * GB
STORM_BYTES = STATE70 * REPLICAS
STORM_S = STORM_BYTES / INGEST
ONCE_S = STATE70 / INGEST
STEP10, EVERY = 10, 100
LOCAL_WRITE_S = PER_NODE_CKPT / NVME4_BW
DRAIN_S = CKPT / PFS_BW
PAUSE_SHARE = LOCAL_WRITE_S / SAVE_INTERVAL_S

# --- Versions (Napkin Math 4.8; table 5) --------------------------------------
SYN_PAYLOAD, SYN_OVERHEAD, SYN_PASSES = 1.0, 0.40, 3
SYN_REST = SYN_PAYLOAD * (1 + SYN_OVERHEAD)
SYN_MOVED = SYN_PAYLOAD * (1 + SYN_OVERHEAD) + SYN_PASSES * SYN_PAYLOAD
RETAINED_CKPTS = SAVES // 100 + 1      # every 100th plus the final (Sec. 4.6.2)
ARCHIVE_TB = (3 + 6) + RETAINED_CKPTS * CKPT / TB

# --- Economics (Sec. 4.8; Systems Perspective 4.3; Napkin Math 4.9) ----------
OBJ_RATE, EGRESS, NVME_RATE = 0.02, 0.09, 0.10   # $/GB-month, $/GB, $/GB-month
DS = 100_000                                      # 100 TB in GB
STORE_YEAR = DS * OBJ_RATE * 12
EGRESS_ONE = DS * EGRESS
EGRESS_TEN = 10 * EGRESS_ONE
GET_FEES = 100e6 / 1000 * 0.0004
GLACIER_RESTORE = DS * 0.02 + EGRESS_ONE
GPU_DAY = 1024 * 2 * 24
IDLE_DAY = GPU_DAY * (0.90 - 0.70)
D50, EPOCHS, RUNS = 50_000, 20, 4
STREAM = D50 * OBJ_RATE * 12 + D50 * EPOCHS * EGRESS * RUNS
STAGE_NVME = D50 * NVME_RATE * 12
STAGE_EGRESS = D50 * RUNS * EGRESS
STAGE = D50 * OBJ_RATE * 12 + STAGE_NVME + STAGE_EGRESS
SAVING = STREAM - STAGE
BREAK_EVEN = (STAGE - D50 * OBJ_RATE * 12) / (D50 * EGRESS * RUNS)

# --- Serving (Sec. 4.9) ------------------------------------------------------
WEIGHTS = PARAMS * 2
LOAD = {"object store 1 GB/s": WEIGHTS / (1 * GB),
        "file system share 4 GB/s": WEIGHTS / (4 * GB),
        "one NVMe 14 GB/s": WEIGHTS / (14 * GB),
        "eight NVMe 7 GB/s": WEIGHTS / 8 / (7 * GB),
        "host DRAM 200 GB/s": WEIGHTS / DRAM_BW}
ROLLOUT = 100 * WEIGHTS
ROLLOUT_OBJ_S = ROLLOUT / (100 * GB)
ROLLOUT_PFS_S = ROLLOUT / PFS_BW


def show():
    p = print
    p(f"GPUs: {GPUS:,}")
    p(f"corpus bytes: {CORPUS / TB:.0f} TB")
    p(f"checkpoint bytes per parameter: {CKPT_BPP} B")
    p(f"checkpoint: {CKPT / GB:,.0f} GB (FP16 {PARAMS * B_FP16 / GB:.0f}, master {PARAMS * B_MASTER / GB:.0f}, Adam {PARAMS * B_ADAM / GB:,.0f})")
    p(f"saves in 30 days: {SAVES:,}")
    p(f"checkpoint writes: {CKPT_TOTAL / PB:.1f} PB")
    p(f"writes over reads: {WRITE_READ_RATIO:,.0f}x")
    p(f"per-node checkpoint: {PER_NODE_CKPT / GB:.2f} GB")
    p(f"file system share per node: {PFS_SHARE / GB:.1f} GB/s")
    p(f"ladder ratios: {R_HBM_DRAM:.1f}, {R_DRAM_NVME:.1f}, {R_NVME_PFS:.1f}")
    p(f"cache 95% hits, 100x miss: E[T] = {ET1:.2f} T_hot, {FRAC1 * 100:.0f}% of hot rate (1/{ET1:.1f})")
    p(f"rate lost to 5% misses at 100x: {(1 - FRAC1) * 100:.0f}%")
    p(f"35 TB job: hit rate {H2 * 100:.0f}%, miss {MISS2:.0f}x slower, {RATE2 / GB:.1f} GB/s, {FRAC2 * 100:.0f}% of local")
    p(f"I/O wall plateau: {PLATEAU:.0f}x; at 16x compute: {SPEEDUP_AT_16:.0f}x")
    p(f"text demand: {TEXT_BW / 1e6:.1f} MB/s")
    p(f"image demand: {IMG_BW / GB:.1f} GB/s ({IMG_SHARE_OF_PFS * 100:.0f}% of 1 TB/s)")
    p(f"image over text: {IMG_TEXT_RATIO:,.0f}x")
    p(f"175B job average: {JOB_AVG_BW / 1e6:.1f} MB/s")
    p(f"1 MB clip over a 116 KB image: {VIDEO_OVER_IMAGE:.1f}x")
    p(f"MLPerf passing closed results: {len(ROWS)}")
    for w, name in WORKLOADS:
        p(f"MLPerf median {name}: A100 {MED[(w, 'A100')]:.2f}, H100 {MED[(w, 'H100')]:.2f} GB/s")
    p(f"MLPerf H100 over A100: {min(GEN):.1f} to {max(GEN):.1f}x")
    r = EXAMPLE_ROW
    p(f"MLPerf example row: {r['submitter']} {r['system']}: {r['num_accelerators']:.0f} accelerators, "
      f"{r['num_hosts']:.0f} hosts, {r['io_GBps']} GB/s, {r['au_mean_pct']}% busy")
    p(f"stall sequential: {SEQ_STEP} ms, {SEQ_STALL * 100:.1f}%")
    p(f"stall overlapped: {OVL_STEP} ms, {OVL_STALL * 100:.0f}%")
    p(f"prefetch: {PREFETCH_SPAN:.1f} windows -> {PREFETCH_DEPTH} batches; local p99 -> {PREFETCH_LOCAL}")
    p(f"prefetch memory per node at depth 5: {PREFETCH_NODE_GB:.1f} GB")
    p(f"GDS CPU-seconds per second: {CPU_BUF:.2f} -> {CPU_GDS:.2f}; cores freed about {CORES_FREED}")
    p(f"file reduction: {FILE_REDUCTION:,.0f}x")
    p(f"PFS aggregate: {PFS_AGG / TB:.0f} TB/s")
    p(f"tail: P(all fast) = {P_ALL_FAST:.3f}; slow steps {P_SOME_SLOW * 100:.0f}%")
    p(f"3FS per storage node: {FS3_PER_NODE_GIB:.1f} GiB/s; TiB over TB {TIB_OVER_TB:.2f}")
    p(f"storm: {STORM_BYTES / TB:,.1f} TB, {STORM_S:,.0f} s = {STORM_S / 60:.1f} min; once {ONCE_S:.1f} s; ratio {STORM_S / ONCE_S:,.0f}x")
    p(f"storm with saves every {EVERY} steps of {STEP10} s: {EVERY * STEP10:,} s of training per {STORM_S:,.0f} s save")
    p(f"local write: {LOCAL_WRITE_S:.2f} s; drain: {DRAIN_S:.2f} s; pause share {PAUSE_SHARE * 100:.2f}%")
    p(f"synthetic: {SYN_REST:.1f} TB at rest, {SYN_MOVED:.1f} TB moved")
    p(f"archive: {RETAINED_CKPTS} retained checkpoints, {ARCHIVE_TB:.1f} TB")
    p(f"object storage 100 TB a year: ${STORE_YEAR:,.0f}; one read ${EGRESS_ONE:,.0f}; ten ${EGRESS_TEN:,.0f}")
    p(f"GET fees 100M files: ${GET_FEES:,.0f}; archive restore ${GLACIER_RESTORE:,.0f}")
    p(f"1,024 GPUs per day: ${GPU_DAY:,.0f}; idle 20 points: ${IDLE_DAY:,.0f}")
    p(f"stream ${STREAM:,.0f}; stage ${STAGE:,.0f} (NVMe ${STAGE_NVME:,.0f}, egress ${STAGE_EGRESS:,.0f}); saving ${SAVING:,.0f}; break-even {BREAK_EVEN:.1f} epochs")
    for k, v in LOAD.items():
        p(f"load 350 GB, {k}: {v:.2f} s")
    p(f"rollout: {ROLLOUT / TB:.0f} TB; object store {ROLLOUT_OBJ_S:.0f} s; file system {ROLLOUT_PFS_S:.0f} s")
    all_fast = 0.99 ** 100  # 100 servers, each slow 1 percent of the time (Sec. 4.3.6)
    p(f"fan-out: all 100 fast {all_fast:.3f}; at least one slow {(1 - all_fast) * 100:.1f}% of steps")


if __name__ == "__main__":
    show()
