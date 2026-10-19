#!/usr/bin/env python3
"""Every derived number in Lecture 15 (Responsible Engineering, Part 1),
computed from sourced inputs.

One block per slide, in deck order. Inputs carry their source as a comment;
each printed line is a value the slide or its note shows, under the label
used there. make_figs.py imports from this file so figures and slides share
one computation. "RE" means the chapter Responsible Engineering in Volume I.

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


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


# ---------------------------------------------------------------------------
# Hook and care chain: Obermeyer et al., Science 366:447 (2019).
# Fig. 1B: Black share of auto-enrolled 17.7 percent, 46.5 percent ranked by
# health. Text: 4.8 against 3.8 active chronic conditions at the 97th
# percentile.
# ---------------------------------------------------------------------------
HOOK_COST, HOOK_HEALTH = 17.7, 46.5
HOOK_MULT = HOOK_HEALTH / HOOK_COST
ILL_BLACK, ILL_WHITE = 4.8, 3.8
ILL_MORE_PCT = (ILL_BLACK / ILL_WHITE - 1) * 100

# ---------------------------------------------------------------------------
# Label choice: Obermeyer et al. 2019, Table 2, Black share of the top 3
# percent by training label.
# ---------------------------------------------------------------------------
LABEL_COST, LABEL_AVOID, LABEL_CHRONIC = 14.1, 21.0, 26.7
LABEL_MULT = LABEL_CHRONIC / LABEL_COST

# ---------------------------------------------------------------------------
# Proxy and goal (illustrative, RE Systems Perspective 15.2): clicks end at
# 1.20 of their start, satisfaction at 0.95.
# ---------------------------------------------------------------------------
CLICKS_END, SAT_END = 1.20, 0.95
PROXY_GAP = CLICKS_END - SAT_END
CLICK_WIN_PCT = (CLICKS_END - 1.0) * 100

# ---------------------------------------------------------------------------
# Gender Shades (Buolamwini and Gebru 2018, Table 4, Face++ error, percent).
# ---------------------------------------------------------------------------
GS_ALL, GS_LM, GS_DM, GS_LF, GS_DF = 10.0, 0.8, 0.7, 9.8, 34.5
GS_MULT = GS_DF / GS_LM

# ---------------------------------------------------------------------------
# Estimate: one group's error rate within +/-1 point at 95 percent
# (RE Napkin Math 15.1): margin = z sqrt(p(1-p)/n), worst case p = 0.5.
# ---------------------------------------------------------------------------
Z95 = 1.96
P_WORST = 0.25            # p(1 - p) at p = 0.5
GROUP_SHARE = 0.01        # the group is 1 percent of users
TARGET_PER_GROUP = 10_000  # the chapter's rounded per-group target


def n_one_group(margin):
    return Z95 ** 2 * P_WORST / margin ** 2


def n_gap_equal_groups(margin):
    """Per-group n so the gap between two equal-size groups has this margin:
    margin = z sqrt(0.25/n + 0.25/n)."""
    return Z95 ** 2 * 2 * P_WORST / margin ** 2


def gap_margin(n_per_group):
    return Z95 * math.sqrt(2 * P_WORST / n_per_group)


N_ONE_1PT = n_one_group(0.01)
RANDOM_TOTAL = TARGET_PER_GROUP / GROUP_SHARE
RANDOM_MULT = RANDOM_TOTAL / TARGET_PER_GROUP

# ---------------------------------------------------------------------------
# Slice C (illustrative gate slices; gap 3.5 points, interval +/-3.5).
# The gate's interval is on a gap between two groups, so its standard error
# adds both groups' variances (standard error of a difference of
# proportions). Equal group sizes assumed.
# ---------------------------------------------------------------------------
SLICE_C_MARGIN = 0.035
N_C_ONE_GROUP_OLD = n_one_group(SLICE_C_MARGIN)       # the old slide's value
N_C_GAP = n_gap_equal_groups(SLICE_C_MARGIN)
N_GAP_1PT = n_gap_equal_groups(0.01)
GAP_MARGIN_AT_9604 = gap_margin(N_ONE_1PT) * 100      # points
GAP_RATIO = N_GAP_1PT / N_C_GAP
SQRT2 = math.sqrt(2)

# ---------------------------------------------------------------------------
# Credit service (constructed, RE Sec. 15.3.2.1): confusion matrices.
# (TP, FN, FP, TN)
# ---------------------------------------------------------------------------
A = (4500, 500, 1000, 4000)
B = (600, 400, 200, 800)


def rates(tp, fn, fp, tn):
    n = tp + fn + fp + tn
    return {"approve": (tp + fp) / n, "tpr": tp / (tp + fn),
            "fpr": fp / (fp + tn), "fnr": fn / (tp + fn), "n": n}


RA, RB = rates(*A), rates(*B)
FNR_MULT = RB["fnr"] / RA["fnr"]

# ---------------------------------------------------------------------------
# Shared threshold (RE figure 2): 16 applicants; errors at each line.
# ---------------------------------------------------------------------------
N_APPL = 16
ERR_LINE1 = (3, 1)        # (Subgroup A, Subgroup B)
ERR_LINE2 = (0, 3)
ACC_LINE1 = (N_APPL - sum(ERR_LINE1)) / N_APPL * 100
ACC_LINE2 = (N_APPL - sum(ERR_LINE2)) / N_APPL * 100

# ---------------------------------------------------------------------------
# COMPAS: ProPublica 2016. FPR 44.85 / 23.45, FNR 27.99 / 47.72 percent,
# PPV 0.63 / 0.59 (Black / White). Base rates from its contingency tables:
# 1,901 of 3,696 and 966 of 2,454 reoffended.
# Chouldechova 2017: FPR = p/(1-p) * (1-PPV)/PPV * (1-FNR).
# ---------------------------------------------------------------------------
COMPAS = {"Black": dict(reoff=1901, n=3696, ppv=0.63, fnr=0.2799, fpr_pub=44.85),
          "White": dict(reoff=966, n=2454, ppv=0.59, fnr=0.4772, fpr_pub=23.45)}
for g, d in COMPAS.items():
    d["p"] = d["reoff"] / d["n"]
    d["odds"] = d["p"] / (1 - d["p"])
    d["prec_term"] = (1 - d["ppv"]) / d["ppv"]
    d["one_minus_fnr"] = 1 - d["fnr"]
    d["fpr"] = d["odds"] * d["prec_term"] * d["one_minus_fnr"]
BASE_BLACK_PCT = COMPAS["Black"]["p"] * 100
BASE_WHITE_PCT = COMPAS["White"]["p"] * 100

# ---------------------------------------------------------------------------
# Price of fairness, Group B (constructed, RE Napkin Math 15.2): half repay,
# repaid +$1,500, default -$3,000; TPR 60 -> 90 percent; FPR rises 20 points.
# ---------------------------------------------------------------------------
REPAY_SHARE, GAIN, LOSS = 0.5, 1500, 3000
TPR0, TPR1, FPR0, FPR_RISE = 0.6, 0.9, 0.2, 0.2
U_BEFORE = TPR0 * REPAY_SHARE * GAIN - FPR0 * (1 - REPAY_SHARE) * LOSS
ADDED_REPAID = (TPR1 - TPR0) * REPAY_SHARE * GAIN
ADDED_DEFAULTS = FPR_RISE * (1 - REPAY_SHARE) * LOSS
NET_CHANGE = ADDED_REPAID - ADDED_DEFAULTS
U_AFTER = U_BEFORE + NET_CHANGE
N_B, N_ALL = 2000, 12000
PER_APPLICANT_ALL = -NET_CHANGE * N_B / N_ALL

# ---------------------------------------------------------------------------
# Frontier (hypothetical, RE figure 3): A at disparity 0.18, B at 0.05 and
# 91 percent accuracy. Curve drawn through the points as
# y = 0.85 + 0.1729 x^0.337 (fitted constants, figure only).
# ---------------------------------------------------------------------------
DISP_A, DISP_B = 0.18, 0.05
FRONTIER_MULT = DISP_A / DISP_B


def frontier(x):
    return 0.85 + 0.1729 * x ** 0.337


# ---------------------------------------------------------------------------
# Feedback (illustrative, RE Napkin Math 15.3): gap 3.0 points at launch,
# 3.3 points one quarter later; 5-point gate; original gap 30 points;
# exploration refits the factor to 0.90.
# ---------------------------------------------------------------------------
GAP0, GAP1, GATE, ORIG_GAP, ALPHA_EXPL = 3.0, 3.3, 5.0, 30.0, 0.90
ALPHA = GAP1 / GAP0
K_CROSS = math.log(GATE / GAP0) / math.log(ALPHA)
K_FAIL = math.ceil(K_CROSS)
GAP_AT_FAIL = GAP0 * ALPHA ** K_FAIL
ORIG_AFTER_YEAR = ORIG_GAP * ALPHA ** 4
DECAY_AT_FAIL = GAP0 * ALPHA_EXPL ** K_FAIL

# ---------------------------------------------------------------------------
# Ranking service (RE Sec. 15.3.4, book scenario): 10M users, 20 items a day,
# 1M-item catalog, 1 percent exploration.
# ---------------------------------------------------------------------------
USERS, ITEMS, CATALOG, EXPLORE = 10_000_000, 20, 1_000_000, 0.01
IMPRESSIONS = USERS * ITEMS
SEEN_PCT = ITEMS / CATALOG * 100
EXPLORE_IMPR = IMPRESSIONS * EXPLORE

# ---------------------------------------------------------------------------
# Next lecture: Strubell et al. 2019, Table 1: 284 t (626,155 lb) for the
# architecture search; car lifetime 57 t (126,000 lb).
# ---------------------------------------------------------------------------
NAS_T, CAR_T = 284, 57
NAS_CARS = NAS_T / CAR_T


if __name__ == "__main__":
    head("Hook (slide 2) and care chain (slide 3)")
    show("ranked by health / ranked by cost", f"{f(HOOK_MULT)}x")
    show("more chronic illnesses at the enrollment line", f"{f(ILL_MORE_PCT)}%")

    head("Label choice (Obermeyer Table 2)")
    show("chronic-illness label / cost label", f"{f(LABEL_MULT)}x")

    head("Proxy and goal")
    show("gap clicks - satisfaction", f(PROXY_GAP, 2))
    show("click win", f"{f(CLICK_WIN_PCT, 0)}%")

    head("Gender Shades")
    show("darker women / lighter men", f"{f(GS_MULT, 0)}x")

    head("Estimate: one group within +/-1 point")
    show("n (one group)", f"{N_ONE_1PT:,.0f}")
    show("random collection", f"{RANDOM_TOTAL:,.0f}")
    show("random / stratified", f"{RANDOM_MULT:.0f}x")

    head("Slice C: the gap between two groups")
    show("old single-group n for +/-3.5", f"{N_C_ONE_GROUP_OLD:,.0f}")
    show("gap n per group for +/-3.5", f"{N_C_GAP:,.0f}")
    show("gap n per group for +/-1", f"{N_GAP_1PT:,.0f}")
    show("gap margin with 9,604 per group (points)", f(GAP_MARGIN_AT_9604, 2))
    show("sqrt(2)", f(SQRT2, 2))
    show("+/-1 over +/-3.5 data ratio", f(GAP_RATIO, 1))

    head("Credit service matrices")
    for g, d in (("A", RA), ("B", RB)):
        show(f"Group {g} approval / TPR / FPR / FNR",
             f"{d['approve']*100:.0f}% / {d['tpr']*100:.0f}% / {d['fpr']*100:.0f}% / {d['fnr']*100:.0f}%")
    show("approval gap", f"{(RA['approve']-RB['approve'])*100:.0f} pts")
    show("TPR gap", f"{(RA['tpr']-RB['tpr'])*100:.0f} pts")
    show("FPR gap", f"{(RA['fpr']-RB['fpr'])*100:.0f} pts")
    show("FNR B / A", f"{FNR_MULT:.0f}x")

    head("Shared threshold")
    show("overall accuracy, line 1", f"{ACC_LINE1:g}%")
    show("overall accuracy, line 2", f"{ACC_LINE2:g}%")

    head("COMPAS and Chouldechova's identity")
    for g, d in COMPAS.items():
        show(f"{g} base rate p", f(d['p'], 3))
        show(f"{g} 1-p", f(1 - d['p'], 3))
        show(f"{g} p/(1-p)", f(d['odds'], 3))
        show(f"{g} 1-PPV", f(1 - d['ppv'], 2))
        show(f"{g} (1-PPV)/PPV", f(d['prec_term'], 3))
        show(f"{g} 1-FNR", f(d['one_minus_fnr'], 3))
        show(f"{g} FPR from identity", f(d['fpr'], 3))
        show(f"{g} published FPR", f"{d['fpr_pub']}%")
    show("base rates (percent, figure)", f"{f(BASE_BLACK_PCT)} / {f(BASE_WHITE_PCT)}")

    head("Price of fairness, Group B")
    show("before", f"${U_BEFORE:,.0f}")
    show("added repaid", f"+${ADDED_REPAID:,.0f}")
    show("added defaults", f"-${ADDED_DEFAULTS:,.0f}")
    show("net change", f"-${-NET_CHANGE:,.0f}")
    show("after", f"${U_AFTER:,.0f}")
    show("across all applicants", f"${PER_APPLICANT_ALL:,.2f}")

    head("Frontier")
    show("A disparity / B disparity", f"{f(FRONTIER_MULT)}x")
    show("curve at B (accuracy)", f(frontier(DISP_B), 3))
    show("curve at A (accuracy)", f(frontier(DISP_A), 3))

    head("Feedback")
    show("factor", f(ALPHA, 2))
    show("ln(5/3)/ln(1.10)", f(K_CROSS))
    show("quarter the gate fails", K_FAIL)
    show("gap in that quarter", f(GAP_AT_FAIL))
    show("original 30-point gap after four quarters", f(ORIG_AFTER_YEAR))
    show("gap with factor 0.90 in that quarter", f(DECAY_AT_FAIL))

    head("Ranking service")
    show("impressions a day", f"{IMPRESSIONS:,}")
    show("share of catalog seen per user", f"{SEEN_PCT:g}%")
    show("exploration impressions a day", f"{EXPLORE_IMPR:,.0f}")

    head("Next lecture")
    show("search / car lifetimes", f(NAS_CARS))
