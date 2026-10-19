#!/usr/bin/env python3
"""Every derived number in Lecture 14b (ML Operations, Part 2), computed from
sourced inputs.

One block per slide, in deck order. Inputs carry their source as a comment;
each printed line is a value the slide or its note shows, under the label
used there. make_figs.py imports from this file so figures and slides share
one computation. "MO" means the chapter ML Operations in Volume I.

Run from this folder: python3 calcs.py
"""

import math


def r(x, nd=1):
    """Round half away from zero (Python's round() is half to even)."""
    q = 10 ** nd
    return math.floor(abs(x) * q + 0.5 + 1e-9) / q * (1 if x >= 0 else -1)


def f(x, nd=1, commas=False):
    v = r(x, nd)
    return f"{v:,.{nd}f}" if commas else f"{v:.{nd}f}"


# ---------------------------------------------------------------------------
# Half the drop needs about four times the evidence (evidence_step*)
# One-sample binomial test, two-sided 95 percent confidence, 80 percent power,
# from 95 percent accuracy (the sizing of Lecture 12b). z values: standard
# normal quantiles. Budget of 1,000 outcomes: MO Napkin Math 14.2.
# ---------------------------------------------------------------------------
ZA, ZB = 1.959964, 0.841621   # z(0.975), z(0.80)
P0 = 0.95


def n_needed(p0, p1):
    return ((ZA * math.sqrt(p0 * (1 - p0)) + ZB * math.sqrt(p1 * (1 - p1)))
            / (p0 - p1)) ** 2


DROPS = [(5, 0.90), (2, 0.93), (1, 0.94)]
NEED = [n_needed(P0, p1) for _, p1 in DROPS]
NEED_HALF_POINT = n_needed(P0, 0.945)


def need_label(n):
    """Slide label: nearest 10 above 1,000, nearest 5 below."""
    return f"{r(n, -1):,.0f}" if n > 1000 else f"{5 * round(n / 5):,.0f}"


# ---------------------------------------------------------------------------
# Estimate: how long until a drop is confirmed? (delay_step*)
# MO Napkin Math 14.2: 1,000 outcomes at 1 per second and at 100 per day.
# Visa card-absent fraud disputes: up to 120 calendar days (Visa Core Rules,
# 18 April 2026, table 11-29).
# ---------------------------------------------------------------------------
BUDGET = 1000
T_FAST_S = BUDGET / 1.0            # seconds at 1 label per second
T_SLOW_D = BUDGET / 100            # days at 100 labels per day
FAST_SLOW_RATIO = T_SLOW_D * 86400 / T_FAST_S
FRAUD_DAYS = 120

# ---------------------------------------------------------------------------
# Label-free tests (rabanser_step*): time to collect n samples at 1 per second
# and at 100 per day. Detection accuracies: Rabanser et al. 2019, Table 2a.
# ---------------------------------------------------------------------------
RAB_N = [10, 100, 1000, 10000]


def dur_s(sec):
    if sec < 60:
        return f"{sec:.0f} s"
    if sec < 3600:
        return f"{f(sec / 60)} min"
    return f"{f(sec / 3600)} h"


def dur_d(days):
    if days < 1:
        return f"{f(days * 24)} h"
    return f"{days:,.0f} day" + ("s" if days != 1 else "")


RAB_TICKS = [(n, dur_s(n / 1.0), dur_d(n / 100)) for n in RAB_N]

# ---------------------------------------------------------------------------
# A month of silent failure (silent_cost): MO Napkin Math 14.3.
# $50M a year, 5 percent drop, 28 days against 1 day, 4 incidents a year.
# ---------------------------------------------------------------------------
REV, DROP, INCIDENTS = 50e6, 0.05, 4
LOSS_MONTHLY = REV * DROP * 28 / 365
LOSS_DAILY = REV * DROP * 1 / 365
LOSS_DIFF = LOSS_MONTHLY - LOSS_DAILY
LOSS_YEAR = LOSS_DIFF * INCIDENTS
LOSS_WHOLE_YEAR = REV * DROP
MONTH_OVER_DAY = LOSS_MONTHLY / LOSS_DAILY

# ---------------------------------------------------------------------------
# K-S critical value 1.36 sqrt(2/n) at 95 percent (Lecture 4b, Napkin Math
# 4.5); promotion gate 0.1 (MO listing 2).
# ---------------------------------------------------------------------------
def ks_crit(n):
    return 1.36 * math.sqrt(2 / n)


KS = {n: ks_crit(n) for n in (1e3, 1e5, 1e6)}
KS_GATE = 0.1
KS_GATE_RATIO = KS_GATE / KS[1e6]

# ---------------------------------------------------------------------------
# False alerts (falsealarm): MO Napkin Math 14.5, one model, 3 deployment
# variants x 50 metrics, each series tested once a day.
# ICU alarms: Drew et al., PLOS ONE 2014: 88.8 percent of 12,671 annotated
# arrhythmia alarms false.
# ---------------------------------------------------------------------------
SERIES = 3 * 50
FALSE_PER_DAY = {a: SERIES * a for a in (0.05, 0.01, 0.001)}
FALSE_PER_WEEK_AT_0001 = FALSE_PER_DAY[0.001] * 7
ICU_FALSE_PCT = 88.8
ICU_TRUE_PCT = 100 - ICU_FALSE_PCT

# ---------------------------------------------------------------------------
# PSI of the age feature (psi_bars): MO table 13.
# PSI = sum over bins of (serving - training) * ln(serving / training)
# (MO equation table; Lecture 4b).
# ---------------------------------------------------------------------------
AGE_BINS = ["18–25", "26–35", "36–45", "46–55", "56–65", "66+"]
AGE_TRAIN = [15, 25, 20, 18, 12, 10]     # percent
AGE_SERVE = [12, 22, 18, 20, 15, 13]     # percent
PSI_TERMS = [(s - t) / 100 * math.log(s / t)
             for t, s in zip(AGE_TRAIN, AGE_SERVE)]
PSI = sum(PSI_TERMS)
AGE_MOVES = [s - t for t, s in zip(AGE_TRAIN, AGE_SERVE)]

# ---------------------------------------------------------------------------
# Retraining interval (saw_step*, tstar, tcost): MO equation 12, Napkin Math
# 14.6, table 17: Q = 1,000,000/day, V = $0.50, A0 = 0.94, gamma = 0.0024/day,
# C = $5,000. Cheaper path C = $1,250 (table 18). Accelerators: four H100s for
# six hours, $96 at the registry's training rate (MO Sec. 14.6.2).
# ---------------------------------------------------------------------------
Q, V, A0, G, CR = 1_000_000, 0.50, 0.94, 0.0024, 5000
SLOPE = Q * V * A0 * G                 # $ lost per day, per day since the run
TWO_C = 2 * CR
TSTAR_SQ = TWO_C / SLOPE
TSTAR = math.sqrt(TSTAR_SQ)
C_CHEAP = 1250
TSTAR_CHEAP = math.sqrt(2 * C_CHEAP / SLOPE)
TSTAR_4X_TRAFFIC = math.sqrt(2 * CR / (4 * SLOPE))


def cost_per_day(T):
    return CR / T + 0.5 * SLOPE * T


CADENCES = [("Daily", 1), ("Every 3 days", 3), ("Weekly", 7), ("Monthly", 30)]
CADENCE_COST = [(lab, T, cost_per_day(T)) for lab, T in CADENCES]
ACC_HOURS = 4 * 6
ACC_COST = 96
ACC_SHARE_PCT = 100 * ACC_COST / CR

# ---------------------------------------------------------------------------
# Scheduled against triggered retraining (rot_step*): MO equation 8,
# A(t) = A0 exp(-gamma t), floor 0.88 (Part 1's four-week slide 94 to 88).
# ---------------------------------------------------------------------------
FLOOR = 0.88
A_AFTER_90 = A0 * math.exp(-G * 90)
T_HIT = math.log(A0 / FLOOR) / G
RESET_1D = A0 * math.exp(-G * (T_HIT + 1))
RESET_10D = A0 * math.exp(-G * (T_HIT + 10))
TRIGGER_FOR_10D = FLOOR * math.exp(G * 10)

# ---------------------------------------------------------------------------
# ML Test Score (mlts_step*): MO Sec. 14.8.1 (Breck et al. 2017): 4 sections
# of 7 tests; the score is the minimum section score. Audit 6, 7, 5, 1 (MO).
# ---------------------------------------------------------------------------
MLTS = {"Data": 6, "Model": 7, "Infrastructure": 5, "Monitoring": 1}
MLTS_TESTS = 4 * 7
MLTS_SCORE = min(MLTS.values())
MLTS_AFTER_TWO_MODEL = min({**MLTS, "Model": MLTS["Model"] + 2}.values())


if __name__ == "__main__":
    def head(t):
        print(f"\n[{t}]")

    def show(label, value):
        print(f"  {label}: {value}")

    head("Half the drop needs about four times the evidence")
    for (d, p1), n in zip(DROPS, NEED):
        show(f"{d}-point drop (95 to {round(p1 * 100)}%), outcomes", f"{n:.1f} -> {need_label(n)}")
    show("2-point / 5-point need", f(NEED[1] / NEED[0], 2))
    show("1-point / 2-point need (halve the drop)", f(NEED[2] / NEED[1], 2))
    show("0.5-point drop, outcomes", f"{NEED_HALF_POINT:,.0f}")
    show("four times the 1-point need", f"{4 * NEED[2]:,.0f}")

    head("Estimate: how long until a drop is confirmed?")
    show("1,000 labels at 1/s, minutes", f(T_FAST_S / 60))
    show("1,000 labels at 100/day, days", f(T_SLOW_D, 0))
    show("slow / fast collection ratio", f"{FAST_SLOW_RATIO:,.0f}")
    show("120 days in 30-day months", f(FRAUD_DAYS / 30, 0))

    head("Label-free tests: collection time per sample count")
    for n, a, b in RAB_TICKS:
        show(f"{n:,} samples", f"{a} at 1/s; {b} at 100/day")

    head("A month of silent failure")
    show("monthly review loss", f"${LOSS_MONTHLY:,.0f}")
    show("daily check loss", f"${LOSS_DAILY:,.0f}")
    show("difference per incident", f"${LOSS_DIFF:,.0f}")
    show("a year at 4 incidents", f"${LOSS_YEAR:,.0f}")
    show("whole-year loss (the common error)", f"${LOSS_WHOLE_YEAR:,.0f}")
    show("monthly / daily", f(MONTH_OVER_DAY, 0))

    head("At a million samples, almost any shift is significant")
    for n, v in KS.items():
        show(f"K-S critical value at {n:,.0f}", f"{v:.5f}")
    show("gate 0.1 / critical value at 1,000,000", f(KS_GATE_RATIO, 1))

    head("A monitor with nothing to find still pages every day")
    for a, v in FALSE_PER_DAY.items():
        show(f"false alerts a day at {a:g}", f"{v:g}")
    show("false alerts a week at 0.001", f(FALSE_PER_WEEK_AT_0001, 2))
    show("ICU alarms true, percent", f(ICU_TRUE_PCT, 1))

    head("Check: the age feature's PSI")
    for b, t, s, mv, term in zip(AGE_BINS, AGE_TRAIN, AGE_SERVE, AGE_MOVES, PSI_TERMS):
        show(f"bin {b}: {t}% to {s}% (moved {mv:+d} points), term", f"{term:.4f}")
    show("bins that moved 3 points", sum(1 for m in AGE_MOVES if abs(m) == 3))
    show("bins that moved 2 points", sum(1 for m in AGE_MOVES if abs(m) == 2))
    show("66+ term: 0.03 x ln(0.13/0.10)", f"{0.03 * math.log(13 / 10):.4f} (ln 1.3 = {math.log(1.3):.3f})")
    show("PSI total", f"{PSI:.4f} -> {PSI:.3f}")

    head("Each cycle without retraining loses a triangle; T*")
    show("QVA0 gamma, $ per day per day", f"{SLOPE:,.0f}")
    show("2C", f"{TWO_C:,.0f}")
    show("2C / QVA0 gamma", f(TSTAR_SQ, 2))
    show("T*, days", f(TSTAR, 2))
    show("T* at C = $1,250, days", f(TSTAR_CHEAP, 2))
    show("T* at four times the traffic, days", f(TSTAR_4X_TRAFFIC, 2))
    for lab, T, c in CADENCE_COST:
        show(f"cost per day, {lab.lower()}", f"${c:,.0f}")
    show("accelerator-hours (4 H100s x 6 h)", ACC_HOURS)
    show("$96 / $5,000, percent", f(ACC_SHARE_PCT, 1))

    head("Detection delay lowers the floor a trigger holds")
    show("accuracy after 90 days", f"{A_AFTER_90:.3f}")
    show("days to 0.88", f(T_HIT, 1))
    show("reset with 1-day delay", f"{RESET_1D:.3f}")
    show("reset with 10-day delay", f"{RESET_10D:.3f}")
    show("trigger to hold 0.88 with 10-day delay", f"{TRIGGER_FOR_10D:.3f}")

    head("An incident ends when it becomes a check")
    show("tests in the ML Test Score", MLTS_TESTS)
    show("score (minimum section)", MLTS_SCORE)
    show("score after two more model tests", MLTS_AFTER_TWO_MODEL)
