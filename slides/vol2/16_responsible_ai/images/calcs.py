"""Every derived number in Lecture 16 (Responsible AI), computed once.

Run:  python3 calcs.py
Each line prints  <slide label> = <value>; the slides and notes copy these
values. Inputs carry their source as a comment. Chapter = Volume II,
Chapter 16, rendered dev text (18 October 2026). make_figs.py imports V.
"""

V = {}


def put(label, value, fmt="{:,.3f}"):
    V[label] = value
    print(f"{label} = {fmt.format(value)}")


# ---------------------------------------------------------------------------
# Hook: Ofqual 2020 (Ofqual, Grading gaps in summer 2020, July 2021)
# ---------------------------------------------------------------------------
# Table 2: candidates with 0 A levels where the CAG exceeded the calculated
# grade: 78,470 of 246,110 (32%).
ofq_zero, ofq_cands = 78_470, 246_110
put("hook: candidates with no grade lowered (%)", 100 * ofq_zero / ofq_cands, "{:.1f}")
put("hook: candidates with at least one grade lowered (%)",
    100 * (ofq_cands - ofq_zero) / ofq_cands, "{:.1f}")

# Table 2, all columns: candidates by the number of A levels whose CAG
# exceeded the calculated grade (0, 1, 2, 3, 4), rounded to the nearest 10.
OFQ_T2 = [78_470, 95_580, 55_730, 16_100, 220]
for k, n in enumerate(OFQ_T2):
    put(f"flows: candidates with {k} lowered (%)", 100 * n / ofq_cands, "{:.1f}")
put("flows: Table 2 counts sum (check, 246,110)", sum(OFQ_T2), "{:,.0f}")

# Table 1: row percentages, CAG (rows) against calculated grade (columns
# A*, A, B, C, D, E, U); "~" (under 0.5%) entered as 0. N per CAG grade.
OFQ_GRADES = ["A*", "A", "B", "C", "D", "E", "U"]
OFQ_N = [91_950, 156_920, 179_700, 145_530, 61_240, 20_260, 1_810]
OFQ_T1 = [[61, 36, 3, 0, 0, 0, 0],
          [1, 56, 39, 3, 0, 0, 0],
          [0, 1, 59, 37, 3, 0, 0],
          [0, 0, 2, 59, 34, 4, 0],
          [0, 0, 0, 5, 59, 31, 4],
          [0, 0, 0, 1, 11, 60, 28],
          [0, 0, 0, 0, 3, 17, 79]]
OFQ_SPLIT = []   # per CAG grade: higher, same, one lower, two or more lower (%)
for i, row in enumerate(OFQ_T1):
    split = (sum(row[:i]), row[i], row[i + 1] if i + 1 < 7 else 0, sum(row[i + 2:]))
    OFQ_SPLIT.append(split)
    put(f"flows: CAG {OFQ_GRADES[i]} higher/same/one lower/two+ lower (%)", 0, "")
    print("   ", split)
lower = sum(n * (s[2] + s[3]) for n, s in zip(OFQ_N, OFQ_SPLIT)) / sum(OFQ_N)
put("flows: entries lowered, from Table 1 (%; report text: 39)", lower, "{:.1f}")
put("flows: Table 1 N sum (check, 457,420)", sum(OFQ_N), "{:,.0f}")

# ---------------------------------------------------------------------------
# Segment 2: the running loan model (Napkin Math 16.1, 16.2, chapter 16.5.1)
# ---------------------------------------------------------------------------
# Group A: 70 approved (40 repaid, 30 defaulted), 30 rejected (5 would
# repay, 25 would default). Group B: 40 approved (30 repaid, 10 defaulted),
# 60 rejected (20 would repay, 40 would default). 100 applicants each.
A = dict(tp=40, fp=30, fn=5, tn=25)
B = dict(tp=30, fp=10, fn=20, tn=40)
for g, d in (("A", A), ("B", B)):
    n = sum(d.values())
    put(f"loan {g}: approval rate", (d["tp"] + d["fp"]) / n, "{:.2f}")
    put(f"loan {g}: TPR", d["tp"] / (d["tp"] + d["fn"]), "{:.3f}")
    put(f"loan {g}: FPR", d["fp"] / (d["fp"] + d["tn"]), "{:.3f}")
    put(f"loan {g}: PPV", d["tp"] / (d["tp"] + d["fp"]), "{:.3f}")
    put(f"loan {g}: base rate (would repay)", (d["tp"] + d["fn"]) / n, "{:.2f}")
put("loan: parity gap", V["loan A: approval rate"] - V["loan B: approval rate"], "{:.2f}")
put("loan: TPR gap", V["loan A: TPR"] - V["loan B: TPR"], "{:.3f}")
put("loan: FPR gap", V["loan A: FPR"] - V["loan B: FPR"], "{:.3f}")
put("loan: PPV gap (B - A)", V["loan B: PPV"] - V["loan A: PPV"], "{:.3f}")
put("loan: disparate impact ratio", V["loan B: approval rate"] / V["loan A: approval rate"], "{:.2f}")
put("loan: average absolute odds difference",
    0.5 * (abs(V["loan: TPR gap"]) + abs(V["loan: FPR gap"])), "{:.3f}")

# ---------------------------------------------------------------------------
# Segment 2: COMPAS, recomputed from ProPublica's released data
# ---------------------------------------------------------------------------
# propublica/compas-analysis, compas-scores-two-years.csv (master), with
# ProPublica's filters: |days_b_screening_arrest| <= 30, is_recid != -1,
# c_charge_degree != 'O', score_text != 'N/A'. "High risk" = Medium or High.
# Counts printed by the scratch script compas_counts.py (2026-10-18):
compas = {
    "Black": dict(tp=1188, fp=641, fn=473, tn=873),
    "White": dict(tp=414, fp=282, fn=408, tn=999),
}
for g, d in compas.items():
    n = sum(d.values())
    put(f"COMPAS {g}: defendants", n, "{:,.0f}")
    put(f"COMPAS {g}: FPR (%)", 100 * d["fp"] / (d["fp"] + d["tn"]), "{:.1f}")
    put(f"COMPAS {g}: PPV (%)", 100 * d["tp"] / (d["tp"] + d["fp"]), "{:.1f}")
    put(f"COMPAS {g}: base rate, reoffended in 2 years (%)", 100 * (d["tp"] + d["fn"]) / n, "{:.1f}")

# ---------------------------------------------------------------------------
# Segment 2: dual thresholds (illustrative scores drawn on the slide)
# ---------------------------------------------------------------------------
# Qualified (would repay) and default scores per group; scenario values
# chosen to reproduce the chapter's figure 5 rates (TPR 75 and 25 percent at
# a single 0.75 cutoff; 75 percent for B at 0.45).
DUAL = {
    "A": dict(qual=[0.93, 0.85, 0.79, 0.60], dflt=[0.70, 0.55, 0.38, 0.22]),
    "B": dict(qual=[0.80, 0.60, 0.50, 0.33], dflt=[0.52, 0.40, 0.28, 0.15]),
}


def tpr_at(g, t):
    q = DUAL[g]["qual"]
    return sum(s >= t for s in q) / len(q)


def fpr_at(g, t):
    q = DUAL[g]["dflt"]
    return sum(s >= t for s in q) / len(q)


put("dual: TPR A at 0.75 (%)", 100 * tpr_at("A", 0.75), "{:.0f}")
put("dual: TPR B at 0.75 (%)", 100 * tpr_at("B", 0.75), "{:.0f}")
put("dual: TPR B at 0.45 (%)", 100 * tpr_at("B", 0.45), "{:.0f}")
put("dual: FPR A at 0.75 (%)", 100 * fpr_at("A", 0.75), "{:.0f}")
put("dual: FPR B at 0.75 (%)", 100 * fpr_at("B", 0.75), "{:.0f}")
put("dual: FPR B at 0.45 (%)", 100 * fpr_at("B", 0.45), "{:.0f}")
# Fairness cost (Napkin Math 16.4): accuracy 85 -> 81 percent.
put("fairness cost: accuracy points", 85 - 81, "{:.0f}")
put("fairness cost: relative drop (%)", 100 * (85 - 81) / 85, "{:.1f}")

# ---------------------------------------------------------------------------
# Segment 2: Obermeyer et al. 2019 (chapter footnote 7): share of Black
# patients among those auto-identified, 17.7 -> 46.5 percent (simulated fix)
# ---------------------------------------------------------------------------
put("Obermeyer: ratio after/before", 46.5 / 17.7, "{:.1f}")

# ---------------------------------------------------------------------------
# Segment 3: two-proportion z-test (chapter 16.5.2), 0.70 vs 0.40
# ---------------------------------------------------------------------------
import math

p1, p2 = 0.70, 0.40


def z_for(n):
    pp = (n * p1 + n * p2) / (2 * n)
    return (p1 - p2) / math.sqrt(pp * (1 - pp) * (2 / n))


put("z: n = 100 per group", z_for(100), "{:.2f}")
put("z: n = 10 per group (20 applications)", z_for(10), "{:.2f}")
pp = (p1 + p2) / 2
n_star = pp * (1 - pp) * 2 * (1.96 / (p1 - p2)) ** 2
put("z: n per group where z = 1.96", n_star, "{:.1f}")
# 95% CI of the gap (unpooled), n = 100 and n = 10
for n in (100, 10):
    se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    put(f"gap 95% CI low, n = {n}", (p1 - p2) - 1.96 * se, "{:.2f}")
    put(f"gap 95% CI high, n = {n}", (p1 - p2) + 1.96 * se, "{:.2f}")

# ---------------------------------------------------------------------------
# Segment 3: Gender Shades (mlsysim literature series gender_shades_audit:
# Buolamwini and Gebru 2018 Table 4; Raji and Buolamwini 2019 Table 1)
# ---------------------------------------------------------------------------
ibm_err = {"lighter male": 0.3, "lighter female": 7.1, "darker male": 12.0, "darker female": 34.7}
for k, e in ibm_err.items():
    put(f"IBM 2017 accuracy, {k} (%)", 100 - e, "{:.1f}")
put("IBM darker-female error ratio to lighter male", 34.7 / 0.3, "{:.0f}")

# ---------------------------------------------------------------------------
# Segment 3: fleet monitoring scenario (chapter 16.5.5)
# ---------------------------------------------------------------------------
qps, groups, per_group_metrics, sample, rec_bytes, alpha = 10_000, 50, 3, 0.01, 200, 0.05
day = qps * 86_400
put("monitor: inferences per day", day, "{:,.0f}")
put("monitor: sampled events per day", day * sample, "{:,.0f}")
put("monitor: GB per day", day * sample * rec_bytes / 1e9, "{:.2f}")
metrics = groups * per_group_metrics
put("monitor: active metrics", metrics, "{:.0f}")
put("monitor: expected false alerts per interval", metrics * alpha, "{:.1f}")
put("monitor: expected false alerts at Bonferroni alpha/150", metrics * alpha / metrics, "{:.2f}")

# ---------------------------------------------------------------------------
# Segment 4: explanation placement (chapter 16.7.2, 16.7.3, 16.11)
# ---------------------------------------------------------------------------
for f in (3, 10, 20):
    put(f"SHAP subsets, {f} features", 2 ** f, "{:,.0f}")
    put(f"SHAP time at 1 us per subset, {f} features (ms)", 2 ** f * 1e-3, "{:,.3f}")
put("retrofit: latency low (ms)", 100 + 50, "{:.0f}")
put("retrofit: latency high (ms)", 100 + 200, "{:.0f}")
put("retrofit: capacity low (x)", (100 + 50) / 100, "{:.1f}")
put("retrofit: capacity high (x)", (100 + 200) / 100, "{:.1f}")
put("storage: TB per month low (1 MB maps)", 1e6 * 30 * 1e6 / 1e12, "{:.0f}")
put("storage: TB per month high (10 MB maps)", 1e6 * 30 * 10e6 / 1e12, "{:.0f}")

# ---------------------------------------------------------------------------
# Segment 4: DP-SGD accuracy (Abadi et al., CCS 2016, sec. 5.2 and Fig. 6)
# ---------------------------------------------------------------------------
mnist = {"none": 98.30, 8: 97, 2: 95, 0.5: 90}        # delta = 1e-5
cifar = {"none": 80, 8: 73, 4: 70, 2: 67}             # delta = 1e-5; "approximately 80%"
for eps in (8, 2, 0.5):
    put(f"MNIST accuracy loss at eps = {eps} (points)", mnist["none"] - mnist[eps], "{:.1f}")
for eps in (8, 4, 2):
    put(f"CIFAR-10 accuracy loss at eps = {eps} (points)", cifar["none"] - cifar[eps], "{:.0f}")

# ---------------------------------------------------------------------------
# Segment 4: unlearning (Napkin Math 16.6)
# ---------------------------------------------------------------------------
full_cost, full_days, shards, req_per_day = 4.6e6, 36, 100, 1000
put("SISA: one-shard cost ($)", full_cost / shards, "{:,.0f}")
put("SISA: one-shard hours", full_days * 24 / shards, "{:.2f}")
put("SISA: full-to-shard cost ratio", full_cost / (full_cost / shards), "{:.0f}")
p_untouched = (1 - 1 / shards) ** req_per_day
put("SISA: P(a shard untouched by a day's 1,000 requests)", p_untouched, "{:.1e}")
put("SISA: expected shards touched per day", shards * (1 - p_untouched), "{:.3f}")
put("SISA: daily batch cost if every touched shard retrains ($)",
    shards * (1 - p_untouched) * full_cost / shards, "{:,.0f}")
# RLHF labeling budget (chapter 16.8.2)
put("RLHF labels: low budget ($)", 50_000 * 0.50, "{:,.0f}")
put("RLHF labels: high budget ($)", 500_000 * 5, "{:,.0f}")
put("RLHF labels: spread (x)", (500_000 * 5) / (50_000 * 0.50), "{:.0f}")

# ---------------------------------------------------------------------------
# Segment 5: automation bias (Napkin Math 16.7)
# ---------------------------------------------------------------------------
ai_sens, human_sens, accept = 0.95, 0.92, 0.7
put("automation bias: accepted misses", (1 - ai_sens) * accept, "{:.3f}")
put("automation bias: best-case team sensitivity (%)", 100 * (1 - (1 - ai_sens) * accept), "{:.1f}")
put("automation bias: accepted misses at 0.6 acceptance", (1 - ai_sens) * 0.6, "{:.3f}")
put("automation bias: accepted misses at 0.8 acceptance", (1 - ai_sens) * 0.8, "{:.3f}")

# ---------------------------------------------------------------------------
# Segment 5: Waymo (mlsysim series waymo_rider_only_miles and
# waymo_vs_human_benchmark; Waymo Safety Impact Data Hub, through June 2026)
# ---------------------------------------------------------------------------
m0, m1 = 22.199, 271.329          # million rider-only miles, 2024-06 and 2026-06
put("Waymo: exposure growth (x)", m1 / m0, "{:.1f}")
inj_rate, inj_bench = 0.6744570049043109, 3.7723613124942825   # per million miles
put("Waymo: injury rate", inj_rate, "{:.2f}")
put("Waymo: human benchmark injury rate", inj_bench, "{:.2f}")
put("Waymo: benchmark / Waymo injury rate", inj_bench / inj_rate, "{:.1f}")
put("Waymo: rate x miles (injury crashes)", inj_rate * m1, "{:.0f}")
put("Waymo: benchmark rate x same miles (expected injury crashes)", inj_bench * m1, "{:.0f}")
# Chapter 16.8.5: 0.1 percent per-request failure at an assumed 1 billion requests/day
put("fleet safety: failures per day at 1e9 requests", 1e9 * 0.001, "{:,.0f}")

# Values quoted in notes
put("hook: days from results (13 Aug) to reversal (17 Aug)", 17 - 13, "{:.0f}")
put("dual: opportunity gap at one cutoff (x)", V["dual: TPR A at 0.75 (%)"] / V["dual: TPR B at 0.75 (%)"], "{:.0f}")
put("z: pooled approval rate", (p1 + p2) / 2, "{:.2f}")
put("automation bias: team gain over AI alone (points)",
    V["automation bias: best-case team sensitivity (%)"] - 100 * ai_sens, "{:.1f}")

if __name__ == "__main__":
    pass
