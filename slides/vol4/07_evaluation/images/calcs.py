#!/usr/bin/env python3
"""Every derived number in Lecture 7 (Closed-Loop Evaluation), Volume IV.

One block per slide, in deck order. Inputs carry their source as a comment
("ch" is the rendered chapter, Closed-Loop Evaluation, with its section or
Napkin Math number); each printed line is a value the slide or its note
shows, under the label used there. make_figs.py imports this file so the
figures and the slides share one computation.

Run from this folder: python3 calcs.py
"""
import math

from scipy import optimize, stats


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


# ---------------------------------------------------------------------------
# Inputs (sources in comments)
# ---------------------------------------------------------------------------
# Hook: NHTSA PE 16-007 closing resume (Jan 2017), Figure 11 and text p. 10
NHTSA_BEFORE, NHTSA_AFTER = 1.3, 0.8        # airbag crashes per million miles
QCS_BEFORE, QCS_AFTER = 0.76, 1.21          # ch War Story 7.1 (QCS 2019), complete-exposure cohort
QCS_EXACT_SHARE = 13                         # ch War Story 7.1: percent of vehicles with exact exposure

# Running case: the cage-door campaign (ch §7.1, §7.7.2, Napkin Math 7.1, 7.2)
N_CLEAN = 20                 # clean runs (illustrative campaign)
ALPHA = 0.05                 # one minus 95 percent confidence (chosen)
RUN_S, RESET_S = 25, 15      # active run and automated reset, s (illustrative)
JAM_RESET_S = 60             # ch §7.2.2: operator clears a jammed pin (illustrative)
OPERATORS = 2                # ch Napkin Math 7.2: safety operators per run
P_ONE_IN_EIGHT = 0.875       # ch §7.6: a policy that fails one run in eight

# Aisle drift (ch §7.3.1, equation 2)
V_BASE = 1.5                 # m/s drive limit
HEADING_DEG = 1.0            # persistent heading offset
CLEARANCE_M = 0.15           # side clearance (100 cm base in a 130 cm aisle)
AISLE_M = 30.0

# Contact (ch §7.3.2)
K_CONTACT = 5e5              # N/m stiff latch interface
PEN_M = 0.002                # 2 mm residual
TRIPWIRE_N = 15.0            # contact tripwire
LATCH_LIMIT_N = 100.0        # structural latch limit (ch §7.9.2)
MEAN_N, P99_N = 12.0, 115.0  # ch §7.9.2: mean and 99th-percentile peak, 100 trials

# Offline illusion, published: Mandlekar et al. 2021, Appendix G (Square PH and
# Transport PH, low-dim BC-RNN, 30% validation split, 3 seeds), success %.
MK = {"Square": (80.7, 2.7), "Transport": (64.0, 0.7)}  # (best checkpoint, lowest val loss)

# Sequential test (ch §7.7.3)
P0, P1 = 0.10, 0.01          # unacceptable and acceptable failure rates (illustrative)
BETA = 0.10                  # allowed false rejection
SPRT_CLEAN_BEFORE_FAIL = 25

# Comparison (ch §7.7.4)
ARM_N, ARM_A, ARM_B = 20, 20, 16
P_BETTER, P_WORSE = 0.90, 0.80
Z_A, Z_B = 1.96, 0.84

# Exposure wall (ch §7.9.1, Napkin Math 7.3)
HOURS_PER_YEAR = 8766        # ch: one rig-year, continuous
RIGS = 100                   # ch: illustrative
SITE_RATE = 1e-4             # ch §7.9.1: industrial requirement per hour
MISSION_S, MISSION_RESET_S, MISSION_M = 360, 120, 540
WHEEL_D_M = 0.266
# Intervention campaign (ch §7.9.1)
INTERVENTIONS, MISSIONS = 38, 100

# RAND RR-1478 Table 1 (via ch figure 5), million miles: zero-failure 95% bound
RAND = {"Fatalities": 275.0, "Injuries": 3.9, "Crashes": 1.6}
RAND_RATE = {"Fatalities": 1.09, "Injuries": 77.0, "Crashes": 190.0}  # per 100M miles
RAND_PRECISION_FATAL = 8800.0
WAYMO_M = 56.7               # Kusano et al. 2025: rider-only miles through Jan 2025

# ---------------------------------------------------------------------------
# Derived values, one function each
# ---------------------------------------------------------------------------


def lower_bound(n, alpha=ALPHA):
    """Clopper-Pearson one-sided success lower limit after n clean runs."""
    return alpha ** (1 / n)


def runs_needed(target, alpha=ALPHA):
    return math.ceil(math.log(alpha) / math.log(target))


def cycle_s():
    return RUN_S + RESET_S


def cp_lower_with_failures(successes, n, alpha=ALPHA):
    """Lower limit p with P(X >= successes | n, p) = alpha (equation 6)."""
    f = lambda p: stats.binom.sf(successes - 1, n, p) - alpha
    return optimize.brentq(f, 1e-9, 1 - 1e-12)


def aisle():
    rad = math.radians(HEADING_DEG)
    t = CLEARANCE_M / (V_BASE * math.sin(rad))
    return rad, rad ** 2, t, V_BASE * t


def sprt():
    up = math.log((1 - P1) / (1 - P0))
    down = math.log(P1 / P0)
    a = math.log((1 - BETA) / ALPHA)
    b = math.log(BETA / (1 - ALPHA))
    n_accept = math.ceil(a / up)
    traced = SPRT_CLEAN_BEFORE_FAIL * up + down
    return up, down, a, b, n_accept, traced


def exposure(rate, alpha=ALPHA):
    return math.log(1 / alpha) / rate


def main():
    head("Hook: NHTSA PE 16-007")
    drop = 1 - NHTSA_AFTER / NHTSA_BEFORE
    show("reported drop (Figure 11 bars), percent", f"{100 * drop:.1f}")
    rise = QCS_AFTER / QCS_BEFORE - 1
    show("complete-exposure cohort change, percent", f"+{100 * rise:.0f}")

    head("Running case: twenty clean runs")
    lb = lower_bound(N_CLEAN)
    show("ln(1/alpha)", f"{math.log(1 / ALPHA):.3f}")
    show("exponent ln(1/alpha)/n", f"{math.log(1 / ALPHA) / N_CLEAN:.4f}")
    show("95% success lower limit, percent", f"{100 * lb:.1f}")
    show("failure upper limit, percent", f"{100 * (1 - lb):.1f}")
    show("rule of three 3/n", f"{3 / N_CLEAN:.2f}")
    show("one in seven? 1/(1-lb)", f"{1 / (1 - lb):.1f}")
    show("P(20 clean | p=0.875), percent", f"{100 * P_ONE_IN_EIGHT ** N_CLEAN:.1f}")
    show("cycle, s", cycle_s())
    show("20 runs, min", f"{N_CLEAN * cycle_s() / 60:.1f}")

    head("Reset latency of a jammed run")
    show("jam reset share of cell time, percent", f"{100 * JAM_RESET_S / (RUN_S + JAM_RESET_S):.1f}")

    head("Denominator filtering: 9 of 10")
    show("with the tenth run, percent", f"{100 * 9 / 10:.0f}")
    show("tenth run dropped, percent", f"{100 * 9 / 9:.0f}")

    head("Offline illusion: Mandlekar et al. 2021, App. G")
    for task, (best, lowval) in MK.items():
        show(f"{task}: best / lowest-val-loss, percent", f"{best} / {lowval}")
        show(f"{task}: ratio best/lowest-val", f"{best / lowval:.0f}")

    head("Aisle drift (equation 2)")
    rad, mse, t, d = aisle()
    show("heading, rad", f"{rad:.5f}")
    show("replay angular MSE, rad^2", f"{mse:.3e}")
    show("time to spend 15 cm, s", f"{t:.1f}")
    show("distance along aisle, m", f"{d:.1f}")
    show("fraction of 30 m aisle", f"{d / AISLE_M:.2f}")

    head("Contact: F = k * delta")
    show("force at 2 mm, N", f"{K_CONTACT * PEN_M:,.0f}")
    show("penetration at 15 N tripwire, mm", f"{1000 * TRIPWIRE_N / K_CONTACT:.2f}")
    show("penetration at 100 N limit, mm", f"{1000 * LATCH_LIMIT_N / K_CONTACT:.2f}")

    head("Clean-run ladder (equation 5)")
    for target in (0.90, 0.99, 0.999):
        n = runs_needed(target)
        show(f"runs for {100 * target:g}%", f"{n:,}")
        show(f"  lower limit at that n, percent", f"{100 * lower_bound(n):.2f}")
        show(f"  cell hours at {cycle_s()} s", f"{n * cycle_s() / 3600:.1f}")
    n999 = runs_needed(0.999)
    show("operator hours for 99.9%", f"{OPERATORS * n999 * cycle_s() / 3600:.1f}")
    show("8-hour shifts", f"{OPERATORS * n999 * cycle_s() / 3600 / 8:.1f}")
    show("ratio 2,995 runs / 20 runs", f"{n999 / N_CLEAN:.0f}")
    show("lower limit after 1,000 clean runs", f"{lower_bound(1000):.4f}")
    show("Bonferroni per-cell runs (alpha/2)", f"{runs_needed(0.999, ALPHA / 2):,}")

    head("Estimate: 99.5 percent at 95 percent")
    show("rule of three 3/(1-0.995)", f"{3 / (1 - 0.995):.0f}")
    n995 = runs_needed(0.995)
    show("exact runs", n995)
    show("cell hours", f"{n995 * cycle_s() / 3600:.1f}")

    head("One failure in twenty (equation 6)")
    show("19 of 20 lower limit, percent", f"{100 * cp_lower_with_failures(19, 20):.1f}")
    show("18 of 20 lower limit, percent", f"{100 * cp_lower_with_failures(18, 20):.1f}")

    head("Sequential test (equations 7, 8)")
    up, down, a, b, n_acc, traced = sprt()
    show("per success", f"+{up:.4f}")
    show("per failure", f"{down:.3f}")
    show("accept boundary", f"{a:.3f}")
    show("reject boundary", f"{b:.3f}")
    show("clean runs to accept", n_acc)
    show("25 clean then 1 failure", f"{traced:.2f}")
    show("fixed 29-run test rejects p_fail=0.01, percent", f"{100 * (1 - (1 - P1) ** 29):.1f}")
    show("Ville false-acceptance bound", f"{ALPHA / (1 - BETA):.3f}")
    show("five 5% looks, at least one false positive, percent", f"{100 * (1 - 0.95 ** 5):.1f}")

    head("Comparing two policies")
    tail = math.comb(20, 4) / math.comb(40, 4)
    show("P(all 4 failures in one arm)", f"{tail:.3f}")
    show("two-sided p", f"{2 * tail:.3f}")
    fisher = stats.fisher_exact([[ARM_A, ARM_N - ARM_A], [ARM_B, ARM_N - ARM_B]])
    show("scipy Fisher two-sided p", f"{fisher.pvalue:.3f}")
    h = 2 * math.asin(math.sqrt(P_BETTER)) - 2 * math.asin(math.sqrt(P_WORSE))
    show("Cohen h", f"{h:.2f}")
    show("runs per arm", f"{2 * ((Z_A + Z_B) / h) ** 2:.0f}")

    head("Exposure wall (equation 9)")
    for rate in (SITE_RATE, 1e-6, 1e-9):
        show(f"hours at {rate:g}/h", f"{exposure(rate):,.0f}")
    show("10^-6: machine-years", f"{exposure(1e-6) / HOURS_PER_YEAR:.0f}")
    show("10^-9: years for 100 rigs", f"{exposure(1e-9) / HOURS_PER_YEAR / RIGS:,.0f}")
    rig = math.log(1 / ALPHA) / HOURS_PER_YEAR
    show("one rig-year bound, per hour", f"{rig:.1e}")
    show("short of 1e-6 by", f"{rig / 1e-6:.0f}")
    show("short of 1e-9 by", f"{rig / 1e-9:,.0f}")
    show("door 99.9% campaign, h", f"{n999 * cycle_s() / 3600:.1f}")
    show("10^-4 hours in years", f"{exposure(SITE_RATE) / HOURS_PER_YEAR:.2f}")

    head("Supervised campaign: 38 interventions in 100 missions")
    hours = MISSIONS * MISSION_S / 3600
    km = MISSIONS * MISSION_M / 1000
    show("active hours", f"{hours:.0f}")
    show("km", f"{km:.0f}")
    show("MTBI, min", f"{hours * 60 / INTERVENTIONS:.1f}")
    show("interventions per km", f"{INTERVENTIONS / km:.2f}")
    lo = stats.chi2.ppf(0.025, 2 * INTERVENTIONS) / 2 / hours
    hi = stats.chi2.ppf(0.975, 2 * (INTERVENTIONS + 1)) / 2 / hours
    show("rate per hour and Garwood 95%", f"{INTERVENTIONS / hours:.1f} ({lo:.1f} to {hi:.1f})")

    head("Road vehicles: RAND against Waymo")
    for k, m in RAND.items():
        check = math.log(1 / ALPHA) / (RAND_RATE[k] / 100)
        show(f"{k}: RAND / recomputed, M miles", f"{m} / {check:.1f}")
    show("fatality zero-failure / Waymo", f"{RAND['Fatalities'] / WAYMO_M:.1f}")
    show("fatality precision / Waymo", f"{RAND_PRECISION_FATAL / WAYMO_M:.0f}")

    head("Mean against peak")
    show("p99 over limit, N", f"{P99_N - LATCH_LIMIT_N:.0f}")
    show("p99 / mean", f"{P99_N / MEAN_N:.1f}")


if __name__ == "__main__":
    main()
