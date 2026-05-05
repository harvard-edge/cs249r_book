# Math Audit Report: `book/quarto/contents/vol1/introduction/introduction.qmd`

## Checked scope

Audited `book/quarto/contents/vol1/introduction/introduction.qmd` for equations, generated numeric examples, unit conversions, complexity/scaling claims, and prose-equation consistency. Direct reasoning only; no Gemini or external verification used.

Checked items include:

- Line 29: `\mlsysstack{30}{30}{30}{30}{30}{30}{30}{30}` count/values.
- Lines 55-107 and 116: Google search and H100/CPU TFLOPS scale values.
- Lines 144-154: 5 percent / 95 percent code split and GPT-3 FP16 memory.
- Lines 163-221: verification-gap calculation for 224 x 224 RGB 8-bit images.
- Lines 243-531: historical numeric examples, STUDENT arithmetic, rule counts, and scaling prose.
- Lines 533-1060: AlexNet relative error reduction, parameter count, GPT-3 parameter/token/FLOP/data examples.
- Lines 1082-1151: GPT-4 GPU-day value and AI performance table.
- Lines 1391-1506: engineering-crux counts, deployment archetype scaling, and cost-scaling equation.
- Lines 1519-1547: degradation examples and degradation equation.
- Lines 1558-1748: iron law, pipelined form, GPT-3 training-time example, energy equation, and RoC equation.
- Lines 1900-2185: efficiency gains, Moore's Law comparisons, and training-compute growth figure.
- Lines 2340-2979: deployment spectrum, lifecycle/pillar counts, Amdahl example, drift example, fallacies, and summary scaling claims.

## Findings

### 1. GPT-3 token count and training FLOPs use different assumptions

- **Lines:** 1038-1047, 1055-1060, 1647-1693, 1702-1716
- **Severity:** Medium
- **Issue:** The prose says GPT-3 consumed approximately 500 billion tokens while reporting the model's generated training compute as about 314 zettaFLOPs. For a dense transformer training estimate, the usual back-of-envelope relation is approximately `6 * parameters * tokens`. With 175 billion parameters and 300 billion training tokens, the compute is `6 * 175e9 * 300e9 = 3.15e23` FLOPs, or about 315 zettaFLOPs, matching the generated value. With 500 billion tokens, the same relation gives `5.25e23` FLOPs, or about 525 zettaFLOPs. The same mismatch propagates to the GPT-3 training-time worked example, which uses the 314 zettaFLOP assumption.
- **Proposed correction:** Either change the prose token claim to "trained on about 300 billion tokens drawn from a roughly 500-billion-token corpus" while keeping the 314 zettaFLOP and 25-day calculations, or keep "500 billion tokens" and update the compute and training-time examples to about 525 zettaFLOPs and about 42 days at 1024 A100s, 312 TFLOPS/GPU, and 45 percent efficiency.

### 2. Deployment archetype span is described as nine orders of magnitude but the table values give about six

- **Lines:** 1448-1473, 1478-1487, 2979
- **Severity:** Medium
- **Issue:** The generated archetype values compare Cloud H100 memory/compute against TinyML ESP32 memory/compute. The memory ratio is `512 GB / 512 KiB = 976,562.5`, about `10^6`, and the compute ratio is `989 TFLOPS / 0.0005 TFLOPS = 1,978,000`, about `2 x 10^6`. The code floors the memory span to `10^5` via `int(math.log10(mem_scaling))`, but even rounded to the nearest order it is about `10^6`, not nine orders. The prose "spans nine orders of magnitude in computational power and memory capacity" is therefore inconsistent with the displayed table. The chapter-connection line repeats "nine orders of magnitude in power and memory"; the listed H100-vs-ESP32 power budgets are 700 W vs 400 mW, only about `1.75 x 10^3`.
- **Proposed correction:** If the table remains single-device H100 vs ESP32, revise prose to "about six orders of magnitude in memory and compute, and about three orders in device power." If the intended claim is data-center fleet vs TinyML, change the table inputs to fleet-scale memory/compute/power and regenerate the span values.

### 3. Moore's Law comparison over the stated 2012-2019 window is overstated

- **Lines:** 1940-1944, 1948-1950, 1982-1992
- **Severity:** Low
- **Issue:** Line 1942 says that over the same 2012-2019 window, Moore's Law delivered roughly 16x transistor scaling. A two-year doubling over seven years gives `2^(7/2) = 11.3x`, not 16x. A 16x scaling corresponds to eight years of two-year doublings. The surrounding prose also alternates between 2012-2019, "over just eight years," and a figure dataset ending at EfficientNet in 2019.4.
- **Proposed correction:** Use "roughly 11x" for 2012-2019, or change the window to 2012-2020 if the intended comparison is 16x. Align the "eight years" phrasing with the selected endpoint.

### 4. "Five orders of magnitude" and "3.4-month doubling" are not mutually consistent over 2012-2018

- **Lines:** 1925-1937, 1944
- **Severity:** Low
- **Issue:** The text says AI training compute increased by roughly five orders of magnitude from 2012 to 2018 while doubling every 3.4 months. Over six years, a 3.4-month doubling gives `72 / 3.4 = 21.18` doublings, or `2^21.18 = 2.4e6`, about 6.4 orders of magnitude. Five orders over six years would imply a slower doubling time of about 4.3 months.
- **Proposed correction:** Either revise "five orders" to "over six orders" for a literal 2012-2018, 3.4-month-doubling calculation, or qualify the window/endpoints more narrowly so the five-order statement and doubling-rate statement are not presented as the same calculation.

### 5. Recommendation-system degradation examples conflict in magnitude

- **Lines:** 1527-1547, 2927-2949
- **Severity:** Low
- **Issue:** Line 1547 says a recommendation system might decline from 85 percent to below 40 percent accuracy over six months, "precisely the degradation the equation predicts," but no divergence or lambda values are supplied to support that large drop. Later, the explicit drift example uses 0.8 percentage points per month for six months, giving `85 - 0.8 * 6 = 80.2` percent. The two examples can both be hypothetical, but they read as the same kind of recommendation-system illustration with very different six-month degradation magnitudes.
- **Proposed correction:** Either soften line 1547 to "could decline substantially under severe drift" and avoid "precisely," or provide concrete lambda/divergence assumptions that yield the below-40-percent result. If the intent is consistency with the later worked example, change the illustrative endpoint to about 80 percent after six months.

## Verified calculations and consistency checks

- Lines 144-146: The 5 percent / 95 percent split is internally consistent.
- Line 154: `175 billion parameters * 2 bytes/parameter = 350 billion bytes`, so 350 GB in decimal FP16 storage is correct.
- Lines 203-219: `224 * 224 * 3 = 150,528` channel values; `log10(256^150,528) = 150,528 * log10(256) = 362,507.5`, so "over 362,508 digits" after formatting is correct.
- Line 221: Verification gap approximation is valid when test coverage is negligible relative to total input space.
- Lines 456-457: STUDENT example gives `2 * (0.2 * 45)^2 = 2 * 9^2 = 162`, correct.
- Lines 562-574 and 583-585: AlexNet relative top-5 error reduction is `(26.2 - 15.3) / 26.2 = 41.6 percent`, which rounds to 42 percent.
- Lines 1112-1151: 2.5 million GPU-days is internally consistent with the generated GPT-4 table value.
- Lines 1500-1506: The cost-scaling equation is acceptable as a rough proportionality if "Hardware Efficiency" means compute delivered per dollar and model size times dataset size is treated as a proxy for total compute.
- Lines 1527-1535: The degradation equation is dimensionally coherent if lambda carries units of accuracy per unit divergence.
- Lines 1558-1572: The iron-law terms are dimensionally seconds: bytes divided by bytes/second, FLOPs divided by FLOPs/second, plus latency.
- Lines 1656-1716: With `O ~= 3.14e23` FLOPs, 1024 A100s at 312 TFLOPS each, and 45 percent efficiency, training time is about 25 days; at 60 percent efficiency, about 19 days, saving about 6 days. This is correct under the 314 zettaFLOP assumption noted in Finding 1.
- Lines 1724-1727: Energy equation is dimensionally joules: bytes times joules/byte plus operations times joules/operation.
- Line 1746: RoC equation is dimensionally accuracy gain per dollar, as stated.
- Lines 1924-1937: `24 / 3.4 = 7.06`, so the "7x faster" growth-gap calculation is correct.
- Lines 2063-2110: AlexNet-to-GPT-4 figure values give `2.0e25 / 1.2e18 = 1.67e7`, so the `10^7` / ten-million-times claim is correct at order-of-magnitude precision.
- Lines 2649-2670: Four deployment contexts and three case studies are counted consistently.
- Lines 2842-2898: Amdahl example is correct: total latency drops from `60 + 45 + 25 = 130 ms` to `60 + 15 + 25 = 100 ms`, an end-to-end improvement of `1 - 100/130 = 23.1 percent`, versus the naive `1 - 1/3 = 66.7 percent`.
- Lines 2927-2949: Drift example is correct: `0.8 * 6 = 4.8` percentage points, and `85.0 - 4.8 = 80.2` percent.
