# Math Audit: `book/quarto/contents/vol2/backmatter/appendix_assumptions.qmd`

Scope: system-assumption constants, napkin math, units, equations, numeric examples, conversion factors, and prose-equation consistency. Direct reasoning only; no Gemini.

## Findings

### 1. H100 electricity cost is overstated by about 3.6x

- **Lines:** 554
- **Severity:** High
- **Issue:** The table caption says electricity at `$0.12/kWh` adds approximately `$0.34/GPU-hour` for an H100 at TDP using `700 W * PUE 1.12 * $0.12`, making it `8.5 percent` of a `$4/GPU-hour` rental cost.
- **Explanation:** Direct arithmetic gives `0.700 kW * 1.12 * $0.12/kWh = $0.09408/GPU-hour`, not `$0.34/GPU-hour`. Relative to `$4/GPU-hour`, this is `0.09408 / 4 = 2.35 percent`, not `8.5 percent`.
- **Proposed correction:** Change the electricity add-on to about `$0.09/GPU-hour` and the rental-cost share to about `2.4 percent`. If the intended value is `$0.34/GPU-hour`, use an electricity price of about `$0.43/kWh` or state the additional assumptions.

### 2. Node-level combined MTTF example does not match its formula

- **Lines:** 439
- **Severity:** Medium
- **Issue:** The caption says a node with `8 GPUs, 8 NICs, 2 PSUs, 1 PCIe switch, and 8 HBM stacks` has combined MTTF `1 / (8/50,000 + 8/150,000 + 2/100,000 + 1/200,000 + 8/200,000) approx 2,700 hours (approx 112 days)`.
- **Explanation:** The stated denominator is `0.00027833 failures/hour`, so the reciprocal is `3,593 hours`, or about `150 days`. The displayed `2,700 hours` is not produced by the formula.
- **Proposed correction:** Change the result to `approx 3,600 hours (approx 150 days)`, or add the missing component terms needed to produce the intended `2,700 hours`.

### 3. MTTF table ordering prose conflicts with the listed values

- **Lines:** 431-439
- **Severity:** Medium
- **Issue:** The caption says the MTTF values are ordered from most failure-prone `(GPU die)` to most reliable `(optical cable)`, but `CABLE_MTTF_HOURS` is `50,000 hours`, tied with the GPU value and lower than NIC, PSU, PCIe switch, HBM, and ToR switch values.
- **Explanation:** With the table constants, optical cable/transceiver is among the most failure-prone entries, not the most reliable. The most reliable listed component is `TOR_SWITCH_MTTF_HOURS = 300,000 hours`.
- **Proposed correction:** Either move cable near GPU and change the prose to note that cable/transceiver failures are frequent at scale, or change the final phrase to `to most reliable (top-of-rack switch)`.

### 4. Non-GPU failure-rate prose is inconsistent with the named components

- **Lines:** 132
- **Severity:** Medium
- **Issue:** The quick calculation says the GPU-only failure estimate does not include `NIC, PSU, or cable failures, which collectively double the rate`.
- **Explanation:** For an 8-GPU-node interpretation, the named non-GPU component rates per 8 GPUs are `8/150,000 + 2/100,000 + 8/50,000 = 0.000233 failures/hour`, compared with the GPU rate `8/50,000 = 0.000160 failures/hour`. Those named components add about `1.46x` the GPU-only rate, making the total about `2.46x` GPU-only, not exactly `2x`. If cables are omitted, NIC+PSU adds only about `0.46x`, and the total is about `1.46x`.
- **Proposed correction:** Replace with a qualified statement such as `which can roughly double or more than double the GPU-only rate depending on the component inventory`, or compute the multiplier from the same component model used in the reliability table.

### 5. AllReduce quick estimate calls peak port bandwidth "effective" despite later warning

- **Lines:** 73, 88-90, 134, 626-628
- **Severity:** Medium
- **Issue:** The quick example uses `INFINIBAND_NDR_BW_GBS = 50 GB/s` and describes it as `effective per port`, producing `2 * 140 GB / 50 GB/s = 5.6 s`. Later, the pitfalls section says effective AllReduce bandwidth is usually only `70--85 percent` of peak link bandwidth.
- **Explanation:** The constant is the raw byte conversion of `400 Gbps / 8 = 50 GB/s`, not a reduced collective bandwidth. Applying the appendix's own `70--85 percent` rule gives an effective bandwidth of `35--42.5 GB/s`, so the same 140 GB gradient AllReduce would take `280/42.5 = 6.6 s` to `280/35 = 8.0 s`.
- **Proposed correction:** Change line 134 to say `50 GB/s peak per port` and keep the `5.6 s` value as an optimistic lower bound, or apply an efficiency factor and report approximately `6.6--8.0 seconds`.

### 6. PUE overhead example is rounded to the wrong megawatt value

- **Lines:** 499
- **Severity:** Low
- **Issue:** The caption says the gap between PUE `1.06` and `1.58` for a `7 MW` IT load translates to `3.4 MW` of additional cooling and infrastructure overhead.
- **Explanation:** The incremental facility power is `(1.58 - 1.06) * 7 MW = 3.64 MW`. Rounded to one decimal place, this is `3.6 MW`, not `3.4 MW`.
- **Proposed correction:** Change `3.4 MW` to `3.6 MW`.

### 7. H100 capacity recap mixes GiB source units with a GB display

- **Lines:** 215, 354
- **Severity:** Low
- **Issue:** `H100Recap.cap_gb_str` converts `H100_MEM_CAPACITY` to decimal `GB`, while the source constant is `80 GiB`. The rendered table will therefore show about `86 GB`, not `80 GiB`.
- **Explanation:** This is dimensionally valid, but it is easy to misread because H100 capacity is commonly discussed using the nominal `80 GB` class, while this source constant is explicitly `80 GiB`. The appendix says all data quantities use decimal SI prefixes, but the hardware constant itself is binary.
- **Proposed correction:** If the goal is to display the source constant, render `80 GiB`. If the goal is decimal SI consistency, keep the conversion but make the table label explicit, e.g. `HBM3 Capacity (decimal equivalent)`.

### 8. Cluster-size prose skips the intermediate tier when describing the failure threshold

- **Lines:** 365-370
- **Severity:** Low
- **Issue:** The caption says `The jump from 256 to 8,192 GPUs` crosses the threshold where failure handling shifts to steady state, while the table includes an intermediate `2,048 GPU` tier.
- **Explanation:** This is not an arithmetic error, but it weakens prose-table consistency. The table's four tiers are `256`, `2,048`, `8,192`, and `100,000` GPUs; the prose describes a direct jump from the first to the third tier.
- **Proposed correction:** Change to `By 8,192 GPUs...` or `The progression from 256 through 8,192 GPUs crosses...`.

## Checks That Look Consistent

- The quick GPU failure-rate examples are arithmetically consistent for GPU-only failures: `8,192 / 50,000 = 0.16384 failures/hour`, or `3.93 failures/day`; `100,000 / 50,000 * 24 = 48 failures/day` (lines 83-85, 132).
- The quick AllReduce payload arithmetic is consistent before bandwidth-efficiency adjustment: `70e9 parameters * 2 bytes = 140 GB`, and `2 * 140 GB / 50 GB/s = 5.6 seconds` (lines 88-90, 134).
- The quick carbon calculation is internally consistent: `10,000 * 700 W * 1.12 = 7.84 MW`; over `720 h`, Quebec at `20 g/kWh` gives about `113 tonnes`, Poland at `820 g/kWh` gives about `4,629 tonnes`, a `41x` ratio (lines 92-95, 136).
- The cluster node counts are consistent at `8 GPUs/node`: `256`, `2,048`, `8,192`, and `100,000` GPUs correspond to `32`, `256`, `1,024`, and `12,500` nodes (lines 365-370).
- The wide-area latency example is consistent: `5,000 km / 200,000 km/s = 0.025 s = 25 ms` one-way (line 417).
- The inter-node bandwidth conversion is consistent: `400 Gbps / 8 = 50 GB/s` (lines 405-406, 609).
- The WUE example is consistent: `10 MW = 10,000 kWh/hour`; at `1.8 L/kWh` this is `18,000 L/hour` and `432,000 L/day`, matching the rounded `430,000 L/day` claim (line 511).
- The carbon-intensity range is consistent: Poland `820 gCO2/kWh` divided by Norway `10 gCO2/kWh` gives an `82x` range (lines 519-526, 641).
- The scaling-efficiency example is consistent: `8,192 * 0.35 = 2,867.2` effective GPUs (line 586).
- The overhead-budget arithmetic is consistent: `5% + 3% + 10% + 5% = 23%`, leaving `77%` useful wall time; `30 days * 0.77 = 23.1 days` (lines 590, 599).
