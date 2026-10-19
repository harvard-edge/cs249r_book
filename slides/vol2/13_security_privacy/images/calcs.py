#!/usr/bin/env python3
"""Every derived number in Lecture 13 (Security and Privacy), computed from
sourced inputs. Each line prints the value and the slide that shows it.
"SP" is the rendered chapter, Security & Privacy (Volume II, Chapter 13).

Run from this folder: python3 calcs.py
"""

import math


def show(slide, label, value):
    print(f"{slide:<34} {label:<52} {value}")


# Recall (Lecture 12 chapter, Napkin Math 12.5): canary 5 percent, cutover 100 percent
show("Recall: canary", "exposure ratio cutover / canary", f"{100 / 5:.0f} times")

# Output shaping (SP Sec. 13.5.1): 1,000 classes, top-5; rounding 6 -> 2 decimals
classes, top_k = 1000, 5
show("Carlini / output contract", "scores per answer, full / top-5", f"{classes / top_k:.0f} times fewer")
show("Carlini / output contract", "share of entries removed", f"{100 * (1 - top_k / classes):.1f}%")
show("Carlini / output contract", "bits removed per class, 4 decimal digits",
     f"{4 * math.log2(10):.1f} bits")

# Napkin Math 13.1 (scenario): 5M queries, $0.001/query, paid tier $50 per 30 days at
# 10K queries/day, lifetime allowance 100K queries, training cost $5K
q, price, tier_q, tier_usd, days_per_month, allow = 5e6, 0.001, 1e4, 50.0, 30.0, 1e5
days = q / tier_q
tier_cost = days * tier_usd / days_per_month
allow_fee = allow / tier_q * tier_usd / days_per_month
life_cost = allow_fee + (q - allow) * price
show("Predict: daily quota", "days of collection on the paid tier", f"{days:,.0f}")
show("Predict: daily quota", "paid-tier bill for the full set", f"${tier_cost:,.0f}")
show("Predict: daily quota", "pay-per-query bill for the full set", f"${q * price:,.0f}")
show("Predict: daily quota", "tier fees during the 100K allowance", f"${allow_fee:,.0f}")
show("Predict: daily quota", "overage billed beyond the allowance", f"${(q - allow) * price:,.0f}")
show("Predict: daily quota", "lifetime-budget bill", f"${life_cost:,.0f}")
show("Predict: daily quota", "paid-tier bill / training cost", f"{tier_cost / 5000:.2f}")

# Carlini et al. 2024 (IEEE S&P): 0.01 percent of LAION-400M
show("Nightshade note", "0.01 percent of 400M images", f"{400e6 * 1e-4:,.0f}")

# Breakdown point (illustrative): 63 honest values of 1.0, one value of 10,000
show("Breakdown note", "mean of 64 with one value 10,000", f"{(63 * 1.0 + 1e4) / 64:.0f}")

# Gather traffic (SP Sec. 13.6.1): N_DP = 64
n = 64
ring = 2 * (n - 1) / n
gather = n - 1
show("Estimate: robust median", "ring AllReduce bytes per worker (x S)", f"{ring:.2f}")
show("Estimate: robust median", "gather-all bytes per worker (x S)", f"{gather}")
show("Estimate: robust median", "ratio gather / ring", f"{gather / ring:.0f} times")
show("Estimate: robust median", "GPUs per replica on 1,024 GPUs", f"{1024 // n}")

# Secure multi-tenancy scenario (SP Systems Perspective 13.2): 1,000 -> 850 tokens/s
show("Focus: Fleet Law", "throughput lost", f"{100 * (1000 - 850) / 1000:.0f}%")

# Side channel (SP Sec. 13.7.4): 5-byte password
show("Hardware questions", "brute-force guesses 256^5", f"{256 ** 5:.2e}")
show("Hardware questions", "byte-by-byte guesses 5 x 256", f"{5 * 256:,}")

# Enclave page cache (SP Sec. 13.8.1): 128 MiB; ResNet-50 FP32 weights 102.4 MB
epc_mb = 128 * 2 ** 20 / 1e6
show("Protected boundary", "128 MiB in MB", f"{epc_mb:.1f}")
show("Protected boundary", "ResNet-50 FP32 share of the EPC", f"{100 * 102.4 / epc_mb:.1f}%")

# Napkin Math 13.2 (assumed overheads)
show("Protected boundary", "FHE latency, 20 ms x 10,000", f"{20 * 10000 / 1000:.0f} s")
show("Protected boundary", "AES latency, 20 + 0.5 ms", f"{20 + 0.5:.1f} ms")

# Napkin Math 13.3: sensitivity $200,000, epsilon 1
sens, eps = 200_000, 1.0
scale = sens / eps
show("Estimate: salary", "Laplace scale on the sum", f"${scale:,.0f}")
show("Estimate: salary", "expected |error| of the mean, 1,000 people", f"${scale / 1000:,.0f}")
show("Estimate: salary", "expected |error| of the mean, 100 people", f"${scale / 100:,.0f}")

# Composition (SP Sec. 13.9.2; Abadi et al. 2016)
show("Composition", "basic bound, ten releases at epsilon 1", f"{10 * 1.0:.0f}")
show("Composition", "strong / accountant, 100 epochs", f"{9.34 / 1.26:.1f} times")
show("Composition", "strong / accountant, 400 epochs", f"{24.22 / 2.55:.1f} times")
show("Estimate: salary (note)", "common error: divide by 1,000 twice", f"${scale / 1000 / 1000:.1f}")
