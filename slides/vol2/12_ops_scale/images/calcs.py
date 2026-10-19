#!/usr/bin/env python3
"""Every derived number in Lecture 12 (ML Operations at Scale), computed from
sourced inputs. Run: python3 calcs.py. Each line prints the value with the
slide or note that shows it. "OPS" is the rendered chapter, ML Operations at
Scale.
"""

import math


def show(label, value):
    print(f"{label:<62} {value}")


# --- Operational load sketch (OPS figure 1 code): alerts 20N, deploy N ln N,
#     conflicts N^2/2, capacity 2,500.
cross = next(n for n in range(1, 501) if 20 * n + n * math.log(n) + 0.5 * n * n >= 2500)
show("load: total crosses capacity at N =", cross)
show("load: conflict pairs at N = 100", 0.5 * 100 ** 2)
show("load: alert units at N = 100", 20 * 100)

# --- Point-in-time join (OPS figure 6): 10 clicks at midnight, 4 by noon.
show("pit: naive / point-in-time", 10 / 4)
# Storage (OPS Sec. 12.4.3): 100 M users, 1,000 features, 8,760 h, 32 B/row.
values = 100e6 * 1000 * 8760
show("pit: feature values in history (trillion)", values / 1e12)
show("pit: bytes (PB)", values * 32 / 1e15)
show("pit: metadata share (24 of 32 B)", 24 / 32)

# --- Feature store scale (OPS Sec. 12.4.6): 1 billion recs/day x 100 features.
show("fstore: lookups per second (million)", 1e9 * 100 / 86400 / 1e6)

# --- Broken-feature cost (OPS Sec. 12.4.7): 8% of $15M/week, 5 days.
loss5d = 0.08 * 15e6 * 5 / 7
show("feature cost: 5-day loss ($K)", loss5d / 1e3)
show("feature cost: 4-hour loss ($K, derived)", loss5d * 4 / 120 / 1e3)
show("feature cost: window ratio 120 h / 4 h", 120 / 4)
show("feature cost: reduction", 1 - 4 / 120)
show("feature cost: CTR change 4.2 -> 3.87", 3.87 / 4.2 - 1)
show("feature cost: score mean change 0.42 -> 0.31", 0.31 / 0.42 - 1)

# --- Exposure (OPS Napkin Math 12.5) and delayed alert (Napkin Math 12.6).
show("exposure: 100% / 5%", 100 / 5)
req = 5000 * 86400
lost = req * 0.005
show("delayed alert: requests per day", req)
show("delayed alert: lost clicks per day", lost)
show("delayed alert: $ per day", lost * 0.50)
show("synthesis: $ per hour at that rate (derived)", lost * 0.50 / 24)

# --- Canary stages (OPS equation 7): n = 10,000; r = 1 M/h.
show("stage: hours at 1%", 10000 / (1e6 * 0.01))
show("stage: minutes at 5%", 10000 / (1e6 * 0.05) * 60)
show("stage: configured schedule hours (2 + .5 + .5 + 1)", 2 + 0.5 + 0.5 + 1)

# --- Sample size (OPS equation 4).
z2 = (1.96 + 0.84) ** 2
show("mde: (1.96 + 0.84)^2", z2)
n1 = z2 * 2 * 0.05 * 0.95 / 0.005 ** 2
n2 = z2 * 2 * 0.01 * 0.99 / 0.0005 ** 2
show("mde: n per variant, 5% baseline, 0.5 pt", n1)
show("mde: n per variant, 1% baseline, 0.05 pt", n2)
show("mde: 20 uncorrected tests, P(any false positive)", 1 - 0.95 ** 20)

# --- TFX load latency (Baylor et al. 2017): 500-1,500 ms vs 75-150 ms.
show("tfx: ratio low ends 500/75", 500 / 75)
show("tfx: ratio high ends 1500/150", 1500 / 150)

# --- False alerts (OPS Napkin Math 12.7, equation 9).
show("alerts: 1,000 x 288 x 0.003 per day", 1000 * 288 * 0.003)
show("alerts: 1,000 x 288 x 0.05 per day", 1000 * 288 * 0.05)
show("alerts: 1% of 14,400 left after dedup", 0.01 * 1000 * 288 * 0.05)
show("alerts: P(>=1 false), p=0.003, M=1000", 1 - 0.997 ** 1000)
show("alerts: P(>=1 false), p=0.05, M=100", 1 - 0.95 ** 100)

# --- SLO vector (OPS figure 12, illustrative).
show("slo: mean of 0.40, 0.60, 1.30, 0.50", (0.40 + 0.60 + 1.30 + 0.50) / 4)
# SRE workbook: burn 14.4 for 1 h of a 30-day (720 h) window.
show("sre: budget spent by burn 14.4 over 1 h", 14.4 * 1 / 720)
show("sre: budget spent by burn 6 over 6 h", 6 * 6 / 720)
show("sre: budget spent by burn 1 over 72 h", 1 * 72 / 720)

# --- Meta outage (Cloudflare): unavailable 15:50 to 21:20 UTC.
show("meta: hours unreachable", (21 + 20 / 60) - (15 + 50 / 60))

# --- Philly (Jeon et al. 2019, Table 6).
show("philly: killed + unsuccessful GPU time (%)", 37.69 + 17.76)
show("philly: killed + unsuccessful jobs (%)", 13.5 + 17.2)

# --- Own against rent, 1,024 H100s (OPS figure 13).
own0 = 128 * 0.350 + 5 + 10
rent80 = 1024 * 4 * 730 * 0.80 / 1e6
rent50 = 1024 * 4 * 730 * 0.50 / 1e6
show("buyrent: owned upfront ($M)", own0)
show("buyrent: rent at 80% ($M/month)", rent80)
show("buyrent: crossing (months)", own0 / (rent80 - 1.5 / 12))
show("buyrent: rent at 50% ($M/month, derived)", rent50)
show("buyrent: crossing at 50% (months, derived)", own0 / (rent50 - 1.5 / 12))

# --- 10,000 GPUs (OPS Napkin Math 12.9).
hw = 1250 * 350e3
cap = hw + 25e6 + 75e6
power = 10000 * 0.7 * 1.1 * 8760 * 0.07
own3 = cap + 3 * (power + 5e6)
rent_yr = 10000 * 4 * 8760 * 0.8
full3 = 10000 * 4 * 8760 * 3
show("10k: hardware ($M)", hw / 1e6)
show("10k: capital ($M)", cap / 1e6)
show("10k: electricity ($M/yr)", power / 1e6)
show("10k: owned 3-year total ($M)", own3 / 1e6)
show("10k: rent per year ($M)", rent_yr / 1e6)
show("10k: rent 3 years ($M)", 3 * rent_yr / 1e6)
show("10k: saving ($M)", (3 * rent_yr - own3) / 1e6)
show("10k: rent every hour for 3 years ($M)", full3 / 1e6)
show("10k: break-even utilization", own3 / full3)
show("10k: hours in three years", 8760 * 3)

# --- TCO (OPS table 31, constructed).
st = [5.0, 2.0, 0.5, 40.0]
co = [150, 400, 80, 350]
show("tco: startup total ($K/month)", sum(st))
show("tco: startup shares train/infer/data/iter", [round(v / sum(st) * 100, 1) for v in st])
show("tco: company total ($K/month)", sum(co))
show("tco: company shares train/infer/data/iter", [round(v / sum(co) * 100, 1) for v in co])
show("tco: INT8 payback (months) 80K / (400K x 0.4)", 80 / (400 * 0.4))
show("tco: training payback (months) 80K / (150K x 0.3)", 80 / (150 * 0.3))
show("platform: BCR at 50 models (50 x 20 x 150 / 120,000)", 50 * 20 * 150 / 120000)
show("platform: BCR at 20 models", 20 * 20 * 150 / 120000)
