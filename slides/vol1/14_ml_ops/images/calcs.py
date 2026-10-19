#!/usr/bin/env python3
"""Every derived number in Lecture 14 (ML Operations, Part 1), computed from
sourced inputs.

One block per slide, in deck order. Inputs carry their source as a comment;
each printed line is a value the slide or its note shows, under the label
used there. make_figs.py imports from this file so figures and slides share
one computation. "MO" means the chapter ML Operations in Volume I.

Run from this folder: python3 calcs.py
"""

import math
from statistics import NormalDist


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


def r(x, nd=1):
    """Round half away from zero (Python's round() is half to even)."""
    q = 10 ** nd
    return math.floor(abs(x) * q + 0.5 + 1e-9) / q * (1 if x >= 0 else -1)


def f(x, nd=1, commas=False):
    v = r(x, nd)
    return f"{v:,.{nd}f}" if commas else f"{v:.{nd}f}"


# ---------------------------------------------------------------------------
# Hook: Zillow bought three times more homes than it sold
# Zillow Group Q3 2021 shareholder letter (Form 8-K, exhibit 99.3, p. 8):
# 9,680 homes purchased, 3,032 sold, 9,790 in inventory, 8,172 under contract.
# Q2 2021 letter: 3,805 bought, 2,086 sold, 3,142 in inventory.
# ---------------------------------------------------------------------------
Q2_BOUGHT, Q2_SOLD, Q2_INV = 3805, 2086, 3142
Q3_BOUGHT, Q3_SOLD, Q3_INV, Q3_CONTRACT = 9680, 3032, 9790, 8172
WRITE_DOWN_M = 304            # $ million, Q3 2021 letter
BOUGHT_PER_SOLD = Q3_BOUGHT / Q3_SOLD
Q3_OVER_Q2_BOUGHT = Q3_BOUGHT / Q2_BOUGHT

# ---------------------------------------------------------------------------
# Wild-Time (MO figure 1): id_mean and ood_mean from Yao et al. 2022, Table 1
# (ERM, Eval-Fix), as stored in mlsysim literature series
# temporal_shift_wildtime.csv. The figure plots these to one decimal.
# ---------------------------------------------------------------------------
WILD = [  # label on slide, training period, later period
    ("Yearbook portraits", 97.99, 79.50),
    ("MIMIC mortality (AUC)", 90.89, 72.89),
    ("MIMIC readmission", 73.00, 61.33),
    ("HuffPost news", 79.40, 70.42),
    ("arXiv papers", 53.78, 45.94),
    ("FMoW satellite", 58.07, 54.07),
]
WILD_DROPS = [(n, a - b) for n, a, b in WILD]

# ---------------------------------------------------------------------------
# Outages: Papasian and Underwood, OpML 2020 slides, "Finding #1" chart
# (read from the chart): outages by ML-ness 0 to 4.
# ---------------------------------------------------------------------------
OUTAGES = [61, 0, 9, 2, 24]

# ---------------------------------------------------------------------------
# Automation break-even: MO Napkin Math 14.1 (80 h pipeline, 4 h a week by
# hand) and its self-check (120 h pipeline, 6 h a week). Upkeep lines (1 h and
# 2 h a week) are the deck's: break-even = build / (manual - upkeep).
# ---------------------------------------------------------------------------
def breakeven(build_h, manual_h, upkeep_h=0.0):
    return build_h / (manual_h - upkeep_h)


# ---------------------------------------------------------------------------
# dvc.lock frame: iterative/example-get-started @ 06ea489, data/data.xml.dvc
# (outs size 14445097 bytes) and dvc.lock (model.pkl size 1958115 bytes).
# ---------------------------------------------------------------------------
DATA_XML_BYTES = 14_445_097
MODEL_PKL_BYTES = 1_958_115

# ---------------------------------------------------------------------------
# Parity: MO equation 2 inputs (Sec. 14.1.1.3): 1 percent skew errors,
# 1,000,000 queries a day, $0.10 an error.
# ---------------------------------------------------------------------------
SKEW, QPD, COST_ERR = 0.01, 1_000_000, 0.10
SKEW_COST_YR = SKEW * QPD * COST_ERR * 365

# ---------------------------------------------------------------------------
# Knight Capital: SEC Release 34-70694, para. 15 to 19: market open 9:30,
# "approximately 45 minutes"; new code on seven of eight servers.
# ---------------------------------------------------------------------------
OPEN_MIN = 9 * 60 + 30
KNIGHT_MIN = 45
SERVERS, UPDATED = 8, 7

# ---------------------------------------------------------------------------
# Argo Rollouts docs, features/analysis.md @ 176a66f: steps setWeight 20, 40,
# 60, 80 with pause 10m after each; successCondition result[0] >= 0.95;
# failureLimit 3 (the run fails when failures exceed the limit).
# ---------------------------------------------------------------------------
ARGO_WEIGHTS = [20, 40, 60, 80]
ARGO_PAUSE_MIN = 10
ARGO_FAILURE_LIMIT = 3

# ---------------------------------------------------------------------------
# A/B sizing: MO equation 4, n = 2 (z_a + z_b)^2 sigma^2 / delta^2, with
# sigma^2 = p (1 - p), two-sided 95 percent confidence, 80 percent power.
# Baseline 5 percent; 2 percent relative lift; 25,000 users per variant.
# ---------------------------------------------------------------------------
P0, REL_LIFT, N_AVAIL = 0.05, 0.02, 25_000
Z_A = NormalDist().inv_cdf(1 - 0.05 / 2)
Z_B = NormalDist().inv_cdf(0.80)
VAR = P0 * (1 - P0)
DELTA = P0 * REL_LIFT
N_NEEDED = 2 * (Z_A + Z_B) ** 2 * VAR / DELTA ** 2
MDE = math.sqrt(2 * (Z_A + Z_B) ** 2 * VAR / N_AVAIL)

# ---------------------------------------------------------------------------
# Node-day: MO Sec. 14.4.5, figure 8, table 11: 100 GB of embedding tables;
# 80 GB per H100; eight accelerators per node; retraining on four for six
# hours. ResNet-50: 25.6 million parameters (MO table 11 basis), FP16 = 2 bytes.
# Canary share 5 percent.
# ---------------------------------------------------------------------------
TABLES_GB, H100_GB, NODE_GPUS = 100, 80, 8
RETRAIN_GPUS, RETRAIN_H = 4, 6
RESNET_PARAMS, FP16_BYTES = 25.6e6, 2
CANARY = 0.05
GPUS_PER_COPY = math.ceil(TABLES_GB / H100_GB)

# ---------------------------------------------------------------------------
# Dual bank: MO Sec. 14.4.4 and table 11: 200 KB model, 4 MB flash.
# ---------------------------------------------------------------------------
KWS_KB, FLASH_MB = 200, 4


def main():
    head("Hook: Zillow bars (p. 2)")
    show("Q3 bought per home sold", f(BOUGHT_PER_SOLD, 2))
    show("Q3 purchases over Q2 purchases", f(Q3_OVER_Q2_BOUGHT, 2))

    head("Wild-Time drops (silent failure)")
    for n, d in WILD_DROPS:
        show(f"drop, {n}", f(d, 1))
    ds = [d for _, d in WILD_DROPS]
    show("range on title", f"{f(min(ds), 0)} to {f(max(ds), 1)} points")

    head("Outages (hidden debt)")
    show("total outages", sum(OUTAGES))
    show("middle ratings 1 to 3", sum(OUTAGES[1:4]))

    head("Automation break-even")
    show("80 / 4 (weeks)", f(breakeven(80, 4), 0))
    show("80 / (4 - 1) (weeks)", f(breakeven(80, 4, 1), 1))
    show("120 / 6 (weeks)", f(breakeven(120, 6), 0))
    show("120 / (6 - 2) (weeks)", f(breakeven(120, 6, 2), 0))

    head("dvc.lock frame")
    show("data.xml (MB)", f(DATA_XML_BYTES / 1e6, 1))
    show("model.pkl (MB)", f(MODEL_PKL_BYTES / 1e6, 1))

    head("Parity: skew cost (equation 2)")
    show("errors a day", f"{SKEW * QPD:,.0f}")
    show("cost a year ($)", f"{SKEW_COST_YR:,.0f}")

    head("Knight timeline")
    end = OPEN_MIN + KNIGHT_MIN
    show("open plus 45 minutes", f"{end // 60}:{end % 60:02d}")
    show("servers without the new code", SERVERS - UPDATED)

    head("Argo Rollouts canary")
    show("weights (percent)", ARGO_WEIGHTS)
    show("total pause before full rollout (min)",
         len(ARGO_WEIGHTS) * ARGO_PAUSE_MIN)
    show("failed measurements that fail the run", ARGO_FAILURE_LIMIT + 1)

    head("A/B estimate (equation 4)")
    show("z (95% two-sided)", f(Z_A, 4))
    show("z (80% power)", f(Z_B, 4))
    show("target rate (percent)", f(100 * (P0 + DELTA), 1))
    show("users per variant", f"{math.ceil(N_NEEDED):,}")
    show("detectable difference with 25,000 (percentage points)", f(100 * MDE, 2))
    show("detectable rate with 25,000 (percent)", f(100 * (P0 + MDE), 2))
    show("detectable relative lift (percent)", f(100 * MDE / P0, 0))
    show("needed over available", f(N_NEEDED / N_AVAIL, 1))

    head("Node-day memory map")
    show("H100s per copy of the tables", GPUS_PER_COPY)
    show("H100s holding tables with a canary", 2 * GPUS_PER_COPY)
    show("retraining accelerator-hours", RETRAIN_GPUS * RETRAIN_H)
    free = NODE_GPUS - 3 * GPUS_PER_COPY
    show("free H100s after serving, standby, canary", free)
    show("free copies", free // GPUS_PER_COPY)
    show("canary share of traffic", f"1/{round(1 / CANARY)}")
    show("ResNet-50 FP16 weights (MB)", f(RESNET_PARAMS * FP16_BYTES / 1e6, 1))

    head("Dual-bank update")
    show("two banks (KB)", 2 * KWS_KB)
    show("share of 4 MB flash (percent)", f(100 * 2 * KWS_KB / (FLASH_MB * 1000), 0))


if __name__ == "__main__":
    main()
