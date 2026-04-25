#!/usr/bin/env python3
"""Queueing hockey-stick curve: M/M/1 response time vs. utilization.

Plots the canonical 1/(1-rho) blowup, with P50 and P99 latency proxies
and a horizontal SLO line. The crossing point is the critical
operating utilization — past that, tail latency runs away.

Renders to $VISUAL_OUT_PATH (set by render_visuals.py).
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT = os.environ.get("VISUAL_OUT_PATH", "cloud-2847.svg")

# M/M/1: mean response time W = (1/mu) / (1 - rho), normalize by 1/mu = 1ms.
# Tail (P99) is roughly mean * ln(100) under exponential service,
# so we use 4.6x mean as a P99 proxy — visually sharper hockey stick.
service_time_ms = 1.0
rho = np.linspace(0.05, 0.97, 200)
mean_latency = service_time_ms / (1 - rho)
p99_latency = mean_latency * 4.6  # ln(100) ~ 4.6 — proxy

# SLO line at 50 ms — lab-typical inference SLO
slo_ms = 50.0
# Where p99 crosses 50 ms?
cross_idx = np.argmax(p99_latency > slo_ms)
crit_rho = rho[cross_idx] if cross_idx > 0 else rho[-1]

fig, ax = plt.subplots(figsize=(6.8, 4.0))
ax.plot(rho, mean_latency, color="#4a90c4", lw=2.0, label="Mean latency (M/M/1)")
ax.plot(rho, p99_latency, color="#c44", lw=2.0, label="P99 latency proxy")
ax.axhline(slo_ms, color="#c87b2a", ls="--", lw=1.2, label=f"SLO = {slo_ms:.0f} ms")
ax.axvline(crit_rho, color="#a31f34", ls=":", lw=1.0)
ax.annotate(
    f"  ρ* ≈ {crit_rho:.2f}\n  past here: SLO violated",
    xy=(crit_rho, slo_ms), xytext=(crit_rho - 0.32, slo_ms * 1.6),
    fontsize=9, color="#a31f34",
    arrowprops=dict(arrowstyle="->", color="#a31f34", lw=0.8),
)

ax.set_xlim(0, 1.0)
ax.set_ylim(0, slo_ms * 3)
ax.set_xlabel("Utilization ρ = λ / μ", fontsize=10)
ax.set_ylabel("Latency (ms, normalized to 1/μ)", fontsize=10)
ax.set_title("Queueing hockey stick: tail latency vs utilization",
             fontsize=11, loc="left")
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.grid(alpha=0.25, ls=":")

# Tighten styling to match book figures
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, format="svg", bbox_inches="tight")
