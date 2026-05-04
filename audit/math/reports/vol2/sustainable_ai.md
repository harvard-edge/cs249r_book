# Math Audit: `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd`

Scope: energy/carbon/power/PUE/sustainability equations, numeric examples, unit conversions, scaling claims, and prose-equation consistency. Direct reasoning only; no Gemini.

## Findings

### 1. Train kinetic-energy comparison is about 10x too small for the stated energy

- **Lines:** 80
- **Severity:** Medium
- **Issue:** The footnote says `10^23` FLOPs at `50 TFLOPs/W` consumes about `2 billion Joules`, equivalent to a `500-ton train traveling at 100 km/h`.
- **Explanation:** The energy calculation is consistent: `10^23 / (50e12 FLOPs/J) = 2e9 J`. But a 500 metric ton train at 100 km/h has kinetic energy `0.5 * 500,000 kg * (27.78 m/s)^2 = 1.93e8 J`, about `0.19 billion J`, not `2 billion J`.
- **Proposed correction:** Change the comparison to a roughly `5,000-ton train traveling at 100 km/h`, or keep the 500-ton train and say the training energy is about `10x` that train's kinetic energy.

### 2. Energy-of-intelligence checkpoint gives a much smaller energy result than nearby GPT-3 prose

- **Lines:** 90, 98-103
- **Severity:** Medium
- **Issue:** The checkpoint asks for energy from `3.14e23` FLOPs at `50 TFLOPs/Watt` and `PUE = 1.1`, while nearby prose says GPT-3 consumed `1,287 MWh`, equivalent to `78 household-years`.
- **Explanation:** The checkpoint arithmetic gives `3.14e23 / 50e12 = 6.28e9 J`; applying PUE gives `6.91e9 J = 1.92 MWh`. Dividing by `10.6 MWh/household-year` gives only `0.18 household-years`, not anything close to the GPT-3 `78 household-years` claim. The discrepancy comes from using an extremely high effective hardware efficiency for the checkpoint compared with the empirical GPT-3 energy figure.
- **Proposed correction:** Either change the checkpoint's efficiency to approximately `0.075 TFLOPs/W` if it is intended to reproduce `1,287 MWh` with PUE 1.1, or explicitly frame the checkpoint as an idealized best-case that is not comparable to the empirical GPT-3 value.

### 3. "Thousands of homes annually" conflicts with the chapter's own household-energy baseline

- **Lines:** 303, 317
- **Severity:** Low
- **Issue:** The chapter says training a single language model can consume electricity equivalent to `thousands of homes annually`, but the footnote immediately above says GPT-3's `1,287 MWh` equals `122 households' annual electricity`.
- **Explanation:** Using the stated `10,500 kWh` per household-year, `1,287,000 / 10,500 = 122.6` household-years. That is hundreds at most, not thousands, for the referenced GPT-3-scale example.
- **Proposed correction:** Change line 317 to `hundreds of homes annually`, or specify that the "thousands" claim refers to larger post-GPT-3 frontier runs and provide the corresponding energy value.

### 4. Energy roofline equation uses `max` where physical energy should add

- **Lines:** 1031-1037
- **Severity:** Medium
- **Issue:** The equation defines total energy as `max(E_compute, E_memory)`.
- **Explanation:** Compute and memory energy are separate physical contributions and should add: `E_total = E_compute + E_memory`. A `max(...)` expression can be useful as a roofline-style approximation for the dominant bottleneck, but it undercounts total energy, especially at the crossover where both terms are equal and `max` is half the sum.
- **Proposed correction:** Define `E_total = E_compute + E_memory`, then define `E_dominant = max(E_compute, E_memory)` if the goal is bottleneck classification. The crossover formula `AI_crossover = e_byte / e_flop` remains valid.

### 5. MatMul and vector energy examples convert picojoules to millijoules incorrectly

- **Lines:** 1061-1081
- **Severity:** High
- **Issue:** Several energy values are off by `1000x` because `pJ` is converted to `mJ` incorrectly.
- **Explanation:** `1 mJ = 10^9 pJ`. For `N = 60`, compute energy is `2 * 60^3 * 10 pJ = 4.32e6 pJ = 0.00432 mJ`, not `4.32 mJ`; memory energy is also `0.00432 mJ`, not `4.32 mJ`. For vector addition with `N = 1000`, compute energy is `1000 * 10 pJ = 1e4 pJ = 0.00001 mJ`, not `0.01 mJ`; memory energy is `12000 * 100 pJ = 1.2e6 pJ = 0.0012 mJ`, not `1.2 mJ`.
- **Proposed correction:** Change the small-matrix values to `0.00432 mJ` and the vector values to `0.00001 mJ` and `0.0012 mJ`. The large-matrix `20 mJ` and `1.2 mJ` values are consistent.

### 6. Operational-carbon notation conflicts with earlier total-energy notation and can double-count PUE

- **Lines:** 1233-1241, 1253-1257
- **Severity:** Medium
- **Issue:** `E_total` is first defined as component energy after multiplying by PUE, then `C_operational = E_total * CI_grid * PUE` says `E_total` is IT equipment energy and multiplies by PUE again.
- **Explanation:** If a reader combines the two equations literally, PUE is applied twice. The text under line 1257 clarifies a different meaning for `E_total`, but that conflicts with the symbol introduced at line 1235.
- **Proposed correction:** Rename the operational-carbon input to `E_IT`, giving `C_operational = E_IT * PUE * CI_grid`, or keep `E_total` as facility energy and write `C_operational = E_total * CI_grid`.

### 7. Training-emissions notebook comment overstates GPU energy by 2.5x

- **Lines:** 1267-1294
- **Severity:** Low
- **Issue:** The notebook header says `64 A100 GPUs` for `14 days` produce `~21,504 kWh GPU energy`, but the code computes a smaller value.
- **Explanation:** With the stated `400 W` A100 TDP, `64 * 400 W * 336 h = 8,601,600 Wh = 8,602 kWh`. Applying `PUE = 1.2` gives `10,322 kWh` facility energy and `4,428 kg CO2`, matching the code's final emissions. The header's `21,504 kWh` would imply about `1,000 W` per GPU.
- **Proposed correction:** Change the header comment to `~8,602 kWh GPU energy -> ~10,322 kWh facility (PUE 1.2) -> ~4,428 kg CO2`.

### 8. Embodied-carbon clean-grid fraction is mislabeled as share of total footprint

- **Lines:** 1500-1508
- **Severity:** Medium
- **Issue:** The text says `108 kg` embodied carbon would represent `52 percent of total emissions` in Quebec.
- **Explanation:** The Quebec operational emissions from the earlier example are `25,805 kWh * 20 g/kWh / 1000 = 516 kg` if using the header's stale facility energy, or `10,322 kWh * 20 g/kWh / 1000 = 206 kg` using the actual code. With the actual code path, `108 / (108 + 206) = 34 percent` of total emissions. The `52 percent` value is `108 / 206`, i.e. embodied as a fraction of operational emissions, not total emissions.
- **Proposed correction:** Change `52 percent of total emissions` to `34 percent of total emissions`, or say `52 percent of operational emissions`.

### 9. Fallacies section mixes grid intensities and carbon results from different examples

- **Lines:** 2847
- **Severity:** Medium
- **Issue:** The text says the 64-A100 14-day run produces `4.4 metric tons CO2` on a `367 g/kWh` US grid and `206 kg CO2` in Quebec at `34.5 g/kWh`, a `21-fold` difference.
- **Explanation:** The `4.4 metric tons` value comes from the earlier `429 g/kWh` example, not `367 g/kWh`: `10,322 kWh * 0.429 = 4,428 kg`. At `367 g/kWh`, emissions are `3,788 kg`. The `206 kg` Quebec value comes from `20 g/kWh`, not `34.5 g/kWh`; using `34.5 g/kWh` gives `356 kg`. The ratio using the line's stated intensities is `367 / 34.5 = 10.6x`, not `21x`.
- **Proposed correction:** Either use the earlier values consistently (`429 g/kWh`, `20 g/kWh`, `4.4 t`, `206 kg`, `21x`) or recompute the prose for `367 g/kWh` and `34.5 g/kWh` (`3.8 t`, `356 kg`, `10.6x`).

### 10. H100 embodied-carbon example conflicts with the earlier amortized example by two orders of magnitude

- **Lines:** 1488-1496, 2851
- **Severity:** Medium
- **Issue:** The amortized example correctly computes `8 H100s * 150 kg/GPU = 1,200 kg` total manufacturing carbon before amortization, but the fallacies section says `64 H100s contribute 10.5 metric tons embodied carbon` for a 14-day run.
- **Explanation:** `64 * 164 kg = 10.5 metric tons` is the full manufacturing carbon of the GPUs, not the amortized contribution of a 14-day training run. If amortized over a 4-year lifetime, the 14-day job gets `64 * 164 * 14 / (4 * 365) = 101 kg`, close to the earlier `108 kg` example, not `10.5 t`.
- **Proposed correction:** If discussing per-job lifecycle accounting, change `10.5 metric tons` to about `0.10 metric tons amortized embodied carbon`. If discussing full fleet procurement carbon, say explicitly that the full manufacturing footprint is being allocated to the job rather than amortized.

### 11. Water-footprint footnote overcounts Olympic pools by about 7.5x

- **Lines:** 2221-2223
- **Severity:** Low
- **Issue:** The footnote says `12 billion liters annually` equals `37,000 Olympic pools`.
- **Explanation:** One Olympic-size pool is about `2.5 million liters`. `12 billion / 2.5 million = 4,800` pools, not `37,000`. The daily figure `34 million L/day` annualizes to `12.4 billion L/year`, which is consistent with the `12 billion` claim.
- **Proposed correction:** Change `37,000 Olympic pools` to about `4,800-5,000 Olympic pools`, or revise the annual water volume if `37,000` pools is intended.

### 12. PUE-gap footnote understates carbon savings from a 0.1 PUE improvement at hyperscale

- **Lines:** 2621-2623
- **Severity:** Low
- **Issue:** The footnote says each `0.1 PUE` improvement at a `100 MW` AI datacenter saves `hundreds of tons of CO2`.
- **Explanation:** If `100 MW` is IT load, a `0.1` PUE reduction saves `10 MW` continuously, or `87,600 MWh/year`. At the chapter's US-average `0.429 kg/kWh`, that is `37,580 metric tons CO2/year`. Even at a very clean `20 g/kWh`, it is `1,752 metric tons/year`.
- **Proposed correction:** Change `hundreds of tons` to `thousands to tens of thousands of tons annually, depending on grid carbon intensity`.

## Checks That Look Consistent

- The GPT-3 carbon-cost calculation is arithmetically consistent: `1,287 MWh = 1,287,000 kWh`, and `1,287,000 * 0.429 = 552,123 kg CO2`, or about `552` one-passenger round-trip flight equivalents at `1,000 kg` each (lines 135-171).
- The geography-of-carbon example is consistent: `10,000 MWh = 10,000,000 kWh`; at `20 g/kWh` this is `200 tonnes`, at `800 g/kWh` this is `8,000 tonnes`, a `40x` ratio (lines 202-216, 282-295).
- The PUE savings notebook is consistent: `2.0 MW * (1.58 - 1.10) * 8760 = 8,410 MWh/year`, worth about `$588,672` at `$70/MWh`; the `30 percent more efficient` prose matches the reduction from `3.16 MW` facility power to `2.20 MW` (lines 898-928).
- The A100 training-emissions code path is internally consistent: `64 * 400 W * 336 h = 8,602 kWh`, `* 1.2 PUE = 10,322 kWh`, and `* 429 g/kWh = 4,428 kg CO2` (lines 1282-1310, 1348-1370).
- The TinyML duty-cycle examples are arithmetically consistent: `15 mW * 20/1000 + 0.033 mW * 980/1000 = 0.33 mW`, and `9000 mWh / 0.616 mW = 14,610 h = 1.7 years` (lines 2071-2079, 2097-2107).
- The cascade inference example is consistent: `0.5 + 0.10 * (10 + 50) = 6.5 mJ/image`, an `89 percent` reduction relative to `60 mJ/image` always-cloud inference (lines 2179-2188).
- The EDP equation is dimensionally consistent for average power: since `E = P * T`, `EDP = E * T = P * T^2` (lines 2603-2609).
