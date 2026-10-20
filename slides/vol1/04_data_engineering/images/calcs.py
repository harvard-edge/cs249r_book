#!/usr/bin/env python3
"""Every derived number in Lecture 4 (Data Engineering, Part 1).

One block per slide. Inputs are typed with their source in a comment; every
value the slide or its note shows is computed here and printed with the
label the slide uses. make_figs.py imports the plotted values from here.

Run from this folder: python3 calcs.py
"""

# ---------------------------------------------------------------------------
# Predict: false wakes a month at 99.9% (Napkin Math 4.1, chapter Sec. 4.3)
# ---------------------------------------------------------------------------
WINDOW_S = 1                      # one 1-second window checked every second (chapter)
DAYS_PER_MONTH = 30               # chapter: "over 30 days"
WINDOWS_PER_MONTH = 60 * 60 * 24 * DAYS_PER_MONTH // WINDOW_S
REJECTION = {"99.9%": 0.999, "99.99%": 0.9999, "99.999%": 0.99999}
FALSE_WAKES = {k: WINDOWS_PER_MONTH * (1 - r) for k, r in REJECTION.items()}
BUDGET_PER_MONTH = 1              # target: below 1 false wake a month (chapter Sec. 4.3)
FPR_NEEDED = BUDGET_PER_MONTH / WINDOWS_PER_MONTH
REJECTION_NEEDED_PCT = 100 * (1 - FPR_NEEDED)
OVER_BUDGET_99999 = FALSE_WAKES["99.999%"] / BUDGET_PER_MONTH

# ---------------------------------------------------------------------------
# Real detectors trade misses for false alarms (Picovoice wake-word benchmark)
# ---------------------------------------------------------------------------
# Miss rate at 1 false alarm per 10 hours, averaged over six keywords:
# github.com/Picovoice/wake-word-benchmark, README "Accuracy", doc/img/summary.png,
# commit 84919190fb18 (plot.py prints each value with '%.1f%%').
PV_FA_PER_HOUR = 0.1              # plot.py: target_false_alarm_per_hour=0.1
PV_MISS_PCT = {"PocketSphinx": 52.0, "Snowboy": 31.9, "Porcupine": 2.7}
HOURS_PER_MONTH = 24 * DAYS_PER_MONTH
PV_FA_PER_MONTH = PV_FA_PER_HOUR * HOURS_PER_MONTH
PV_VS_BUDGET = PV_FA_PER_MONTH / BUDGET_PER_MONTH

# ---------------------------------------------------------------------------
# Your turn: what does a $150K data budget buy? (Napkin Math 4.2)
# ---------------------------------------------------------------------------
BUDGET_USD = 150_000
SHARE = {"labeling": 0.60, "storage and processing": 0.25, "governance": 0.15}
PRICE_PER_LABEL = 0.10
REVIEW_OVERHEAD = 0.20
LABELING_USD = BUDGET_USD * SHARE["labeling"]
COST_PER_LABEL = PRICE_PER_LABEL * (1 + REVIEW_OVERHEAD)
LABELS = LABELING_USD / COST_PER_LABEL
STORAGE_USD = BUDGET_USD * SHARE["storage and processing"]
GOVERNANCE_USD = BUDGET_USD * SHARE["governance"]

# ---------------------------------------------------------------------------
# Proving one false wake a month: rule of three (chapter Sec. 4.11)
# ---------------------------------------------------------------------------
RULE_OF_THREE = 3                 # 95% upper bound with zero events is ~3/T (Hanley 1983)
HOURS_NEEDED = RULE_OF_THREE * HOURS_PER_MONTH / BUDGET_PER_MONTH
SAMPLE_RATE_HZ = 16_000           # chapter: 16 kHz
BYTES_PER_SAMPLE = 2              # 16-bit PCM
EVAL_GB = HOURS_NEEDED * 3600 * SAMPLE_RATE_HZ * BYTES_PER_SAMPLE / 1e9

# ---------------------------------------------------------------------------
# Finer labels multiply the work (chapter Sec. 4.6.1)
# ---------------------------------------------------------------------------
MASK_W, MASK_H = 1920, 1080
BOXES, COORDS = 10, 4
MASK_ENTRIES = MASK_W * MASK_H
BOX_ENTRIES = BOXES * COORDS
MASK_VS_BOXES = MASK_ENTRIES / BOX_ENTRIES
MASK_HOURS = 1.5                  # Cityscapes, more than 1.5 h per image (Cordts et al. 2016)
FRAMES = 1_000_000                # note's question
WORK_HOURS_PER_YEAR = 2000        # full-time year (chapter's person-year basis)
MASK_TOTAL_HOURS = FRAMES * MASK_HOURS
MASK_PERSON_YEARS = MASK_TOTAL_HOURS / WORK_HOURS_PER_YEAR

# ---------------------------------------------------------------------------
# Labeling costs thousands of training runs (chapter Sec. 4.6.2, tables 3, 4)
# ---------------------------------------------------------------------------
LABEL_BUDGET_USD = 100_000
IMAGENET_TRAIN = 1_281_167        # ImageNet-1k training images (Russakovsky et al. 2015)
EPOCHS = 90
IMG_PER_S_PER_GPU = 3200          # chapter: 3,200 img/s per accelerator
GPUS = 8
GPU_HOUR_USD = (2, 4)             # table 3: cloud GPU training hour $2-4
TRAIN_HOURS = IMAGENET_TRAIN * EPOCHS / (IMG_PER_S_PER_GPU * GPUS) / 3600
TRAIN_USD = tuple(TRAIN_HOURS * GPUS * p for p in GPU_HOUR_USD)
RATIO = (LABEL_BUDGET_USD / TRAIN_USD[1], LABEL_BUDGET_USD / TRAIN_USD[0])

# ---------------------------------------------------------------------------
# Agreement beyond chance: Cohen's kappa (chapter Sec. 4.6.3)
# ---------------------------------------------------------------------------
P_NOT_SPAM = 0.9
P_E = P_NOT_SPAM ** 2 + (1 - P_NOT_SPAM) ** 2
P_O_EXAMPLE = 0.85                # note: a raw 85% agreement
KAPPA_EXAMPLE = (P_O_EXAMPLE - P_E) / (1 - P_E)

# ---------------------------------------------------------------------------
# Predict: can active learning fit a $50K budget? (Napkin Math 4.6)
# ---------------------------------------------------------------------------
POOL = 10_000_000
AL_BUDGET = 50_000
AL_LABEL_USD = 0.50
SCORE_USD = 0.01
RANDOM_LABELS = 1_000_000         # chapter: about a million labels by random sampling
RANDOM_USD = RANDOM_LABELS * AL_LABEL_USD
SCORE_ALL_USD = POOL * SCORE_USD
FILTERED = 500_000
SCORE_FILTERED_USD = FILTERED * SCORE_USD
AL_LABELS = (AL_BUDGET - SCORE_FILTERED_USD) / AL_LABEL_USD

# ---------------------------------------------------------------------------
# Forced alignment labeled 23.4M clips (chapter Sec. 4.6.6, Mazumder 2021)
# ---------------------------------------------------------------------------
MSWC_CLIPS = 23_400_000
S_PER_LABEL = 10
MSWC_HOURS = MSWC_CLIPS * S_PER_LABEL / 3600
MSWC_PERSON_YEARS = MSWC_HOURS / WORK_HOURS_PER_YEAR

# ---------------------------------------------------------------------------
# Speech Commands split (TensorFlow input_data.py, train.py defaults)
# ---------------------------------------------------------------------------
VAL_PCT, TEST_PCT = 10, 10
TRAIN_PCT = 100 - VAL_PCT - TEST_PCT

# ---------------------------------------------------------------------------
# A real dataset card: Speech Commands v0.02 (Hugging Face card, sha 57ba463)
# ---------------------------------------------------------------------------
SC_V2 = {"train": 84_848, "validation": 9_982, "test": 4_890}
SC_V2_TOTAL = sum(SC_V2.values())
SC_V2_TEST_PCT = 100 * SC_V2["test"] / SC_V2_TOTAL


if __name__ == "__main__":
    p = print
    p("Predict / false wakes")
    p(f"  windows a month: {WINDOWS_PER_MONTH:,}")
    for k, v in FALSE_WAKES.items():
        p(f"  {k} rejection: {v:,.0f} false wakes a month")
    p(f"  99.9% per day: about {FALSE_WAKES['99.9%'] / DAYS_PER_MONTH:.0f}")
    p(f"  99.999% over budget: {OVER_BUDGET_99999:.0f} times")
    p(f"  needed false-positive rate: 1 in {WINDOWS_PER_MONTH:,} = {FPR_NEEDED:.2e}")
    p(f"  needed rejection: {REJECTION_NEEDED_PCT:.5f}%")
    p("Picovoice benchmark")
    p(f"  1 false alarm per 10 h = {PV_FA_PER_MONTH:.0f} false alarms a {HOURS_PER_MONTH}-h month")
    p(f"  benchmark operating point vs budget: {PV_VS_BUDGET:.0f} times looser")
    p("Data budget")
    p(f"  labeling: ${LABELING_USD:,.0f}; per label ${COST_PER_LABEL:.2f}; labels {LABELS:,.0f}")
    p(f"  storage and processing ${STORAGE_USD:,.0f}; governance ${GOVERNANCE_USD:,.0f}")
    p("Rule of three")
    p(f"  hours needed: {HOURS_NEEDED:,.0f} h; at 16 kHz 16-bit: {EVAL_GB:.0f} GB")
    p("Label entries")
    p(f"  mask: {MASK_ENTRIES:,} entries; ten boxes: {BOX_ENTRIES}; ratio {MASK_VS_BOXES:,.0f}")
    p(f"  1M masks at 1.5 h: {MASK_TOTAL_HOURS:,.0f} h = {MASK_PERSON_YEARS:,.0f} person-years")
    p("Labeling vs training")
    p(f"  ResNet-50 run: {TRAIN_HOURS:.2f} h on {GPUS} GPUs; ${TRAIN_USD[0]:.2f} to ${TRAIN_USD[1]:.2f}")
    p(f"  ratio: {RATIO[0]:,.0f} to {RATIO[1]:,.0f} times")
    # Slide: inputs are $20 to $40 estimates, so the ratio is rounded to hundreds.
    p(f"  ratio, rounded to hundreds (slide): {round(RATIO[0], -2):,.0f} to {round(RATIO[1], -2):,.0f} times")
    p("Kappa")
    p(f"  p_e = {P_E:.2f}; kappa at p_o = {P_O_EXAMPLE}: {KAPPA_EXAMPLE:.2f}")
    p("Active learning")
    p(f"  random: ${RANDOM_USD:,.0f}; score all: ${SCORE_ALL_USD:,.0f}; "
      f"score 500K: ${SCORE_FILTERED_USD:,.0f}; labels: {AL_LABELS:,.0f}")
    p("MSWC")
    p(f"  by hand at 10 s: {MSWC_HOURS:,.0f} h = {MSWC_PERSON_YEARS:.1f} person-years")
    p("Speech Commands")
    p(f"  which_set split: {TRAIN_PCT}/{VAL_PCT}/{TEST_PCT} train/validation/test (%)")
    p(f"  v0.02 card: {SC_V2_TOTAL:,} clips in splits; test {SC_V2_TEST_PCT:.1f}%")
