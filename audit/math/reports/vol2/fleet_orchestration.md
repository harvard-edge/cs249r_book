# Math Audit: `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd`

Scope: scheduling/orchestration/resource-utilization equations, numeric examples, unit conversions, scaling claims, queueing/packing math, cost examples, autoscaling arithmetic, and prose-equation consistency. Direct reasoning only; no Gemini.

## Findings

### 1. Idle-GPU economics row for 60 percent utilization is inverted

- **Lines:** 503-508
- **Severity:** High
- **Issue:** For a 10,000-GPU cluster at 60 percent utilization, productive GPUs are `10,000 * 0.60 = 6,000` and idle GPUs are `4,000`, not 4,000 productive and 6,000 idle. The stated waste, `$288,000/day`, is therefore the 40-percent-utilization case, not the 60-percent-utilization case.
- **Explanation:** At `$2/GPU-hour`, daily idle waste at 60 percent utilization is `4,000 * $2 * 24 = $192,000/day`. At 80 percent utilization, the stated `$96,000/day` is correct. Moving from 60 percent to 80 percent therefore saves `$96,000/day`, or about `$35.0M/year`, not `$192,000/day` or `$70M/year`.
- **Proposed correction:** Change the 60-percent row to "6,000 GPUs productive, 4,000 idle = $192,000/day wasted" and the improvement row to "$96,000/day, or $35M/year."

### 2. Availability-zone spot interruption probability is off by two orders of magnitude

- **Lines:** 1268
- **Severity:** High
- **Issue:** The prose says that spreading 1,024 GPUs across 3 availability zones where each AZ has a 5 percent hourly interruption probability makes the probability of losing more than 128 GPUs in a single hour less than 0.1 percent.
- **Explanation:** If each AZ-level interruption reclaims that zone's allocation, each AZ holds about `1024/3 = 341` GPUs, already more than 128. The probability of at least one AZ interruption in an hour is `1 - 0.95^3 = 14.3%`, not `<0.1%`.
- **Proposed correction:** Either change the event model to independent small per-instance interruptions and compute the corresponding binomial probability, or keep the AZ-level model and state that losing one full AZ is about a 14 percent hourly risk under the given 5 percent/AZ/hour assumption.

### 3. Autoscaling SLO-violation estimate undercounts the overload window

- **Lines:** 1503-1524, 1538-1542
- **Severity:** Medium
- **Issue:** The notebook reports `3,600` SLO-violating requests by multiplying peak excess demand (`30 QPS`) by an assumed `120 s` average excess duration. That does not match the stated ramp and readiness times.
- **Explanation:** Capacity is 50 QPS. Traffic crosses 50 QPS at `t = 15 s`; new replicas are ready at `t = 195 s`. Excess requests are the area above capacity: ramp portion from 15 to 60 seconds is `(0 + 30)/2 * 45 = 675` requests, and the post-ramp portion from 60 to 195 seconds is `30 * 135 = 4,050` requests. Total is `4,725` requests, not `3,600`.
- **Proposed correction:** Replace the approximation with the integrated overload area (`~4,725` requests), or explicitly state a different traffic shape/dropped-request model that justifies the 120-second multiplier.

### 4. Topology placement example mixes 64-GPU and 8-GPU scenarios

- **Lines:** 765-808, 865-873
- **Severity:** Medium
- **Issue:** The figure caption and plotted title describe a 1 GB gradient sync on 64 GPUs, but the callout notebook introduces the same latency values as an AllReduce across 8 GPUs.
- **Explanation:** AllReduce latency and effective bandwidth depend on participant count and topology. The code also labels effective bandwidth as `2.0 / time`, implying a 2 GB transfer basis for a "1 GB gradient sync" (reasonable for a ring-style `~2x` byte model, but unstated). Mixing 8 and 64 GPUs makes the 4.8x result look more generally derived than it is.
- **Proposed correction:** Use one scenario consistently. For example, change the callout to "64 GPUs" and note that effective bandwidth labels use an approximate `2x` AllReduce byte volume for a 1 GB tensor, or change the figure caption/title to 8 GPUs.

### 5. Preemption-churn calculation is ambiguous and internally inconsistent

- **Lines:** 1794-1796
- **Severity:** Medium
- **Issue:** A single preemption event is modeled as 15 minutes lost work + 20 minutes reload + 10 minutes warmup = 45 minutes, and the `$96` cost for 64 GPUs is correct (`64 * 0.75 h * $2/h`). But 12 such events per day waste `12 * 45 min = 9 hours`, not 8.5 hours.
- **Explanation:** The statement "over 35 percent of total capacity" is only true for the affected 64-GPU allocation over a day (`9/24 = 37.5%`), not for an arbitrary cluster's total capacity. As written, "cluster averaging 12 preemptions per day" reads cluster-wide, where the denominator is unspecified.
- **Proposed correction:** Change to "9 hours, or 37.5 percent of the affected 64-GPU job's daily allocation" or define the cluster-wide denominator and compute percentage capacity loss in GPU-hours.

### 6. Spot price prose conflicts with the rendered price

- **Lines:** 134, 1250
- **Severity:** Low
- **Issue:** The text says spot instances are offered at 60 to 70 percent discounts, but the rendered example price is `$0.50` versus `$2.00` on demand, a 75 percent discount.
- **Explanation:** `$0.50 / $2.00 = 25%` of on-demand price, so the discount is `75%`.
- **Proposed correction:** Either set the example spot price to `$0.60-$0.80` for a 60-70 percent discount, or change the prose to "60 to 75 percent" or "about 75 percent in this example."

### 7. "Uniform workload" is used for a coefficient of variation of 1

- **Lines:** 331-357
- **Severity:** Low
- **Issue:** The queueing example labels `C_s = 1` as a "uniform workload." In queueing theory, `C_s = 1` is the coefficient of variation of an exponential service-time distribution, not a uniform/deterministic one.
- **Explanation:** The arithmetic is correct: at `rho = 0.8`, `rho/(1-rho) * (1+C_s^2)/2` gives `4x` for `C_s = 1` and `20x` for `C_s = 3`. The issue is the prose label, which may confuse readers about what distribution is being modeled.
- **Proposed correction:** Rename the baseline to "exponential/moderate-variance workload" or set `C_s` to a value appropriate for a narrow/uniform duration distribution and recompute the baseline multiplier.

## Checks That Look Consistent

- Lines 171-196 and 201-203: `10,000 * $2 * 24 = $480,000/day`; 30 percent idle waste is `$144,000/day`; annual waste is about `$52.6M`; improving productive utilization from 50 percent to 80 percent gives `10,000 * (0.80 - 0.50) / 0.50 = 6,000` equivalent GPUs relative to the 50-percent baseline.
- Lines 226-239 and 245-254: 1,024 stranded GPUs at `$2/GPU-hour` cost `$2,048/hour`, consistent with the deadlock callout.
- Lines 294: `4096 * 0.001 / 365 = 0.0112` GPU failures/day, reasonably rounded to `0.01`.
- Lines 361-379: The 64-node fragmentation example is internally consistent: `64 * 8 = 512` GPUs; a 6-GPU job on an 8-GPU node strands 2 GPUs, so effective capacity is `6/8 = 75%`.
- Lines 737-739: NDR InfiniBand conversion is consistent if `400 Gbps` is the link rate: `400/8 = 50 GB/s`.
- Lines 984-1004: The elastic-vs-rigid example is arithmetically consistent. Elastic work during the first 8 hours is `0.25*2 + 0.50*2 + 1.00*4 = 5.5` full-scale hours, leaving `18.5` hours of full-scale work and a total completion time of `26.5` hours.
- Lines 1063-1106: The elastic scaling decision is internally consistent: Strategy 1 is `24.0` epochs, Strategy 2 is `64.0` epochs, and Strategy 3 is `4*1.8 + (24 - 4 - 10/60)*3.2 = 70.67` epochs, about 10 percent better than waiting.
- Lines 1116-1210: The idle-cost figure arithmetic is correct: at 10,000 GPUs and 50 percent utilization, annual waste is `10,000 * 0.5 * $2 * 8,760 = $87.6M`; moving from 50 percent to 70 percent saves `$35.04M/year`.
- Lines 1298-1308: The spot-vs-on-demand worked example is arithmetically consistent: on-demand is `$344,064`; spot total is `512 * 359.8 * $0.70 = $128,872`; savings are `$215,192`, about 62.5 percent, and added wall time is `23.8/336 = 7.1%`.
- Lines 1599-1638 and 1641-1650: GPU sharing ROI is internally consistent assuming A100 memory renders as 80 GB: exclusive fleet is `100 + 10*8 = 180` GPUs; shared fleet is `50 + 80 = 130`; savings are 50 GPUs or 27.8 percent; annual value is `50 * $2 * 8,760 = $876,000`.
- Lines 1688-1719: The diurnal GPU shift calculation is internally consistent: static active GPUs are `150 + 600 = 750`; dynamic active GPUs are `150 + 600 + 130 = 880`; annual value is `130 * $2 * 8,760 = $2.28M`, rendered as `$2.3M`.
- Lines 1758: The over-commitment example is correct: `70% * 1.3 = 91%`, below physical capacity.
- Lines 1808: 500 GPUs idle for three weeks at `$2/GPU-hour` costs `500 * 21 * 24 * $2 = $504,000`, consistent with "roughly $500,000."
- Lines 1934-1950 and 2003-2027: The 60-to-84 percent utilization improvement is a 40 percent relative throughput increase. Expressed as capacity at the old 60-percent efficiency, achieving 840 productive GPUs would require `840/0.60 = 1,400` physical GPUs, i.e. 400 additional GPUs.
- Lines 2091: The M/M/1 wait multiplier statement is correct: `rho/(1-rho)` is `9x` at `rho = 0.90` and `19x` at `rho = 0.95`.
