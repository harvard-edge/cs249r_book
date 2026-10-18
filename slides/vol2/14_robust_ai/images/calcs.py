#!/usr/bin/env python3
"""Every derived number in Lecture 14 (Robust AI), computed from sourced inputs.

Run from this folder: python3 calcs.py
Each line prints the value exactly as the slide or note shows it, with its
slide label. "RA" is the rendered chapter, Robust AI.
"""

import math
from statistics import NormalDist


def show(label, value):
    print(f"{label:<58} {value}")


# --- Tempe timeline (RA Sec. 14.2.2; NTSB HAR-19/03) -----------------------
recognized_s = 1.2   # collision recognized, seconds before impact
hold_s = 1.0         # action suppressed without alert
show("Tempe: alert time before impact (s)", f"{recognized_s - hold_s:.1f}")

# --- Fleet silent corruption (RA Sec. 14.3.1 self-check; p assumed) --------
p = 1e-4             # silent error probability per device-hour (assumed)
show("SDC: P(>=1 per hour), N = 1,000", f"{1 - (1 - p) ** 1_000:.3f}")
show("SDC: P(>=1 per hour), N = 10,000", f"{1 - (1 - p) ** 10_000:.2f}")
show("SDC: 1 - e^-1", f"{1 - math.exp(-1):.2f}")

# --- Mean-shift test (RA Napkin Math 14.1) ---------------------------------
sigma, n_req, base, obs = 0.3, 1000, 0.50, 0.55
se = sigma / math.sqrt(n_req)
z = (obs - base) / se
show("Mean test: standard error", f"{se:.4f}")
show("Mean test: standard errors", f"{z:.1f}")

# --- Fraud PSI (RA table 3) ------------------------------------------------
contrib = {"USA": 0.0118, "UK": 0.0021, "Germany": 0.0007, "France": 0.0036, "Other": 0.0470}
psi = sum(contrib.values())
show("Fraud: PSI, user country", f"{psi:.3f}")
show("Fraud: Other's share of PSI (%)", f"{100 * contrib['Other'] / psi:.0f}")
# KS critical value (RA Sec. 14.4.2 step 3): c(alpha) = sqrt(-ln(alpha/2)/2)
alpha = 0.01
c = math.sqrt(-math.log(alpha / 2) / 2)
crit = c * math.sqrt((10_000 + 10_000) / (10_000 * 10_000))
show("Fraud: KS coefficient c(0.01)", f"{c:.2f}")
show("Fraud: KS critical value", f"{crit:.3f}")
show("Fraud: KS statistic / critical", f"{0.089 / crit:.1f}")
# KL closed form (RA Sec. 14.4.2 step 2)
mp, sp_, mq, sq = 4.2, 1.1, 4.5, 1.3
kl = math.log(sq / sp_) + (sp_ ** 2 + (mp - mq) ** 2) / (2 * sq ** 2) - 0.5
show("Fraud: KL divergence, amount", f"{kl:.3f}")

# --- Calibration (Guo et al. 2017, Table 1) --------------------------------
show("Calib: ResNet-110 / LeNet 5 ECE, CIFAR-100", f"{16.53 / 4.85:.1f}")

# --- Selective prediction load (illustrative) ------------------------------
rate, coverage = 10_000, 0.90
show("Selective: requests/s to fallback", f"{rate * (1 - coverage):,.0f}")
risk_at_90 = 0.012 + 0.10 * coverage ** 6     # the illustrative curve in make_figs.py
show("Selective: error at 90% coverage (%, illustrative)", f"{100 * risk_at_90:.1f}")

# --- Conformal rank (RA Sec. 14.4.4) ---------------------------------------
n_cal, a = 1000, 0.1
show("Conformal: (n + 1)(1 - alpha)", f"{(n_cal + 1) * (1 - a):.1f}")
show("Conformal: rank", f"{math.ceil((n_cal + 1) * (1 - a))}")

# --- Self-consistency (RA Sec. 14.8) ---------------------------------------
k = 5
show("Self-consistency: decode work multiple, k = 5", f"{k}")

# --- PGD-7 work model (RA Napkin Math 14.2, listing 1) ---------------------
show("PGD-7: passes per batch", f"{1 + 7}")
show("FGSM listing: passes per batch (1 attack + 2x update)", f"{1 + 2}")

# --- Four-cell penalty (RA Systems Perspective 14.1) -----------------------
show("Four cells: naive gap (points)", f"{76 - 50}")

# --- RobustBench best 2024 (MLSysIM; 75.28 robust, 93.6 clean) -------------
show("RobustBench: best 2024 fails on (%)", f"{100 - 75.28:.0f}")

# --- Certified radius (RA Sec. 14.6.1.1) -----------------------------------
phi = NormalDist().inv_cdf(0.999)
show("Certified: Phi^-1(0.999)", f"{phi:.2f}")
show("Certified: R = 0.5 * Phi^-1(0.999)", f"{0.5 * phi:.1f}")
d = 32 * 32 * 3
show("Certified: pixels in a 32x32x3 image", f"{d:,}")
show("Certified: l2 length of 8/255 box corner", f"{8 / 255 * math.sqrt(d):.2f}")

# --- BadNets (Gu et al. 2017, Table 4) -------------------------------------
show("BadNets: clean accuracy drop (points)", f"{90.0 - 89.3:.1f}")

# --- Bit flip (RA figure 14) -----------------------------------------------
import struct  # noqa: E402
bits = struct.unpack(">I", struct.pack(">f", 0.042))[0]
bad = struct.unpack(">f", struct.pack(">I", bits ^ (1 << 30)))[0]
good = struct.unpack(">f", struct.pack(">I", bits))[0]
show("Bit flip: corrupted value", f"{bad:.1e}")
show("Bit flip: scale, log2", f"{round(math.log2(bad / good))}")
