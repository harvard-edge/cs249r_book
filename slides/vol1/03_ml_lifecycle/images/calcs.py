#!/usr/bin/env python3
"""Every derived number in Lecture 3 (ML Lifecycle), computed from sourced inputs.

One block per slide, in deck order. Inputs carry their source as a comment;
each printed line is a value the slide or its note shows, under the label
used there. make_figs.py imports the inputs below so figures and slides share
one computation. "Chapter" means the rendered ML Lifecycle chapter.

Run from this folder: python3 calcs.py
"""

import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
# Hook (Gulshan et al., JAMA 2016, abstract and Fig. 2; Beede et al., CHI 2020,
# as cited in chapter Sec. 3.1.2)
LAB_SENS, LAB_SPEC = 97.5, 93.4          # EyePACS-1, high-sensitivity point
LAB_HISPEC_SENS, LAB_HISPEC_SPEC = 90.3, 98.1   # EyePACS-1, high-specificity point
UNGRADABLE, SUBMITTED = 393, 1838        # Beede et al. 2020

# Day-153 scenario (chapter Sec. 3.1.2)
MODEL_MB = 4000                          # "needs 4 GB"
RESERVED_MB = 500                        # clinic IT reserves about 500 MB
DAY_REQUEST, DAY_DISCARD = 1, 153
DAYS_PER_MONTH = 365.25 / 12

# Real tool output: torchvision 0.28.0 parameter counts, produced by running
# data/count_params.py (output saved in data/count_params_output.txt).
FP32_BYTES = 4                           # bytes per FP32 weight


def param_counts():
    out = {}
    for line in (DATA / "count_params_output.txt").read_text().splitlines():
        m = re.match(r"(\w+)\s+([\d,]+) parameters", line)
        if m:
            out[m.group(1)] = int(m.group(2).replace(",", ""))
    return out


# Example 3.1 repair schedule (chapter): days per task, in sequence
REPAIR_DAYS = [("Revise preparation", 1), ("Train new weights", 2),
               ("Repeat validation", 2), ("Package the release", 1)]

# CrowdFlower 2016 survey, chapter figure 4 (share of respondents)
SURVEY = [("Cleaning and organizing data", 60), ("Collecting datasets", 19),
          ("Mining data for patterns", 9), ("Other", 5),
          ("Refining algorithms", 4), ("Building training sets", 3)]

# Napkin Math 3.1 (chapter)
WEEKS = 26
LARGE_START, LARGE_GAIN = 95.0, 0.15
SMALL_START, SMALL_GAIN = 90.0, 0.1
SMALL_EFFECTIVE_RUNS = 100               # chapter's effective experiments
CEILING = 99.0
HOURS_PER_WEEK = 24 * 7

# Accuracy trap (chapter Sec. 3.5.1; illustrative population of 100)
POP, SICK = 100, 10

# 224-pixel demonstration (chapter Sec. 3.7.2; photo measured for the slide)
IMAGENET_INPUT = 224
PHOTO_HEIGHT_PX = 857                    # Hao et al. photo, square crop side
LESION_PX = (8, 12)                      # exudate widths measured on the photo

# FDA De Novo DEN180001 (IDx-DR), decision summary pp. 6-7
FDA_TP, FDA_SICK = 173, 198              # mtmDR+ correctly identified
FDA_TN, FDA_HEALTHY = 556, 621           # mtmDR- with "not detected"
FDA_ANALYZED, FDA_COMPLETED = 819, 892
FDA_IMAGEABLE, FDA_GRADED = 819, 852
FDA_FLOOR_SENS, FDA_FLOOR_SPEC = 85.0, 82.5   # prespecified thresholds

# WILDS (Koh et al., ICML 2021), Table 1, Camelyon17-wilds, ERM average acc
WILDS_ID, WILDS_OOD = 93.2, 70.3

# Napkin Math 3.2 (chapter)
PATIENTS_DAY, PHOTOS, PHOTO_MB = 100, 4, 5
UPLINK_MBIT_S = 2
SHIFT_H = 8
SUMMARY_KB = 10

# Napkin Math 3.3 (chapter)
CLINICS, PATIENTS_PER_CLINIC_DAY, DAYS_YEAR = 500, 50, 365
CLOUD_PER_IMAGE = 0.01
CLOUD_NETWORK = 45_000
EDGE_DEVICE = 500
EDGE_MAINT = 25_000
EDGE_PER_IMAGE = 0.001

# Monitoring cadences (chapter Sec. 3.11.1), converted to minutes
CADENCE_DAYS = {"Minute": 1 / 1440, "Day": 1, "Week": 7,
                "Month": 365 / 12, "Quarter": 365 / 4}


def cadence_minutes():
    return {k: v * 1440 for k, v in CADENCE_DAYS.items()}


# Lecture timing: minutes from the speaker notes ([N min] tags in the .tex)
TEX = HERE.parent / "03_ml_lifecycle.tex"
OPTIONAL_TITLES = ["ML behavior can change when only the data changes",
                   "Cycle time sets how many ideas a team can test",
                   "Edge costs more up front", "Check: true or false?"]
REVEAL_SAVINGS = 4        # three activities run as instructor-led reveals
SHORT_SAVINGS = 3 + 2 + 1  # Stage 6 photo only and 26-week answer given (3),
                           # 224-pixel claim only (2), FDA step 1 only (1)


def note_minutes():
    import re as _re
    tex = TEX.read_text()
    total = sum(float(m) for m in _re.findall(r"\\(?:mlsys)?note\{\[(\d+(?:\.\d+)?) min\]", tex))
    opt = 0.0
    for t in OPTIONAL_TITLES:
        i = tex.index(t)
        m = _re.search(r"\\note\{\[(\d+(?:\.\d+)?) min\]", tex[i:])
        opt += float(m.group(1))
    return total, opt


# ---------------------------------------------------------------------------
def main():
    head("Title note: timing")
    total, opt = note_minutes()
    show("notes total", f"{total:.1f} min")
    show("optional frames", f"{opt:.1f} min")
    show("short path", f"{total - opt - REVEAL_SAVINGS:.1f} min")
    show("50-minute path before pacing", f"{total - opt - REVEAL_SAVINGS - SHORT_SAVINGS:.1f} min")

    head("Predict: what went wrong in the clinics?")
    show("ungradable share", pct(UNGRADABLE / SUBMITTED, 1))

    head("Day-153 failure")
    show("months of work discarded",
         f"{(DAY_DISCARD - DAY_REQUEST) / DAYS_PER_MONTH:.1f}")

    head("Memory: model against reservation; torchvision counts")
    show("model / reservation", f"{MODEL_MB / RESERVED_MB:.0f}x")
    for name, n in param_counts().items():
        show(f"{name} FP32 weights", f"{n * FP32_BYTES / 1e6:.1f} MB")

    head("Diagnose: redo chain (Example 3.1)")
    show("days in sequence", sum(d for _, d in REPAIR_DAYS))

    head("Time sinks (CrowdFlower 2016)")
    show("data work (cleaning + collecting)", f"{SURVEY[0][1] + SURVEY[1][1]}%")

    head("Napkin Math 3.1: iteration")
    show("large runs", WEEKS)
    show("large final accuracy", f"{LARGE_START + WEEKS * LARGE_GAIN:.1f}%")
    show("small possible runs", f"{WEEKS * HOURS_PER_WEEK:,}")
    raw = SMALL_START + SMALL_EFFECTIVE_RUNS * SMALL_GAIN
    show("small uncapped accuracy", f"{raw:.0f}%")
    show("small capped accuracy", f"{min(raw, CEILING):.0f}%")
    show("ideas ratio", f"{SMALL_EFFECTIVE_RUNS / WEEKS:.1f}x")

    head("Accuracy trap")
    show("always-normal accuracy", pct((POP - SICK) / POP, 0))
    show("always-normal sensitivity", pct(0 / SICK, 0))

    head("Stage 3: 224 pixels")
    s = IMAGENET_INPUT / PHOTO_HEIGHT_PX
    show("scale 224/857", f"{s:.2f}")
    show("lesion widths at 224",
         f"{LESION_PX[0] * s:.1f} to {LESION_PX[1] * s:.1f} px")

    head("FDA De Novo DEN180001 (IDx-DR)")
    show("observed sensitivity 173/198", pct(FDA_TP / FDA_SICK))
    show("observed specificity 556/621", pct(FDA_TN / FDA_HEALTHY))
    show("analyzable fraction 819/892", pct(FDA_ANALYZED / FDA_COMPLETED, 0))
    show("imageability 819/852", pct(FDA_IMAGEABLE / FDA_GRADED))
    show("lab minus field sensitivity (97.5 vs 87.4 observed)",
         f"{LAB_SENS - 100 * FDA_TP / FDA_SICK:.1f} points")

    head("WILDS Camelyon17 (Table 1)")
    show("ID minus OOD", f"{WILDS_ID - WILDS_OOD:.1f} points")

    head("Napkin Math 3.2: upload")
    day_mb = PATIENTS_DAY * PHOTOS * PHOTO_MB
    mb_s = UPLINK_MBIT_S / 8
    secs = day_mb / mb_s
    show("daily volume", f"{day_mb:,} MB")
    show("uplink", f"{mb_s} MB/s")
    show("upload time", f"{secs:,.0f} s = {secs / 3600:.1f} h")
    show("share of shift", pct(secs / 3600 / SHIFT_H, 1))
    show("one photo", f"{PHOTO_MB / mb_s:.0f} s")
    summary_mb = PATIENTS_DAY * SUMMARY_KB / 1000
    show("traffic reduction", f"{day_mb / summary_mb:,.0f}x")

    head("Napkin Math 3.3: cloud against edge")
    images = CLINICS * PATIENTS_PER_CLINIC_DAY * DAYS_YEAR
    cloud_inf = images * CLOUD_PER_IMAGE
    cloud = cloud_inf + CLOUD_NETWORK
    edge_inf = images * EDGE_PER_IMAGE
    edge_run = EDGE_MAINT + edge_inf
    capex = CLINICS * EDGE_DEVICE
    show("images per year", f"{images:,}")
    show("cloud inference", f"${cloud_inf:,.0f}")
    show("cloud per year", f"${cloud:,.0f}")
    show("edge up front", f"${capex:,.0f}")
    show("edge inference", f"${edge_inf:,.0f}")
    show("edge per year", f"${edge_run:,.0f}")
    show("savings per year", f"${cloud - edge_run:,.0f}")
    show("payback", f"{capex / (cloud - edge_run):.1f} years")

    head("Monitoring cadences")
    cm = cadence_minutes()
    for k, v in cm.items():
        show(f"{k} in minutes", f"{v:,.0f}")
    show("orders of magnitude, minute to quarter",
         f"{math.log10(cm['Quarter'] / cm['Minute']):.1f}")


if __name__ == "__main__":
    main()
