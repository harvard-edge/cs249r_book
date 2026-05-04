# Math Audit: `book/quarto/contents/vol1/ml_systems/ml_systems.qmd`

## Checked scope

Audited the chapter for equations, numeric examples, unit conversions, complexity/scaling claims, and prose-equation consistency by direct reasoning. I checked the deployment-spectrum tables, light-speed latency calculation, power/frequency scaling, memory-wall growth-rate equation, iron-law and bottleneck equations, ResNet-50 cloud/mobile bottleneck example, TCO examples, bandwidth/locality examples, battery/thermal examples, TinyML energy claims, decision-framework braking-distance example, Amdahl's Law example, and fallacy/pitfall quantitative claims.

## Findings

### High severity

- Lines 4200-4202: The training-data scaling example has two arithmetic errors. The dataset increases from 50K samples to 1M samples, which is 20x more data, not 5x. Accuracy increases from 95 percent to 96.5 percent, which is +1.5 percentage points, not a 0.3 percent gain.
  - Proposed correction: Change to "a 1.5 percentage-point gain for 20x more data" or change the example values so that the stated 0.3 percent / 5x relationship is true.

- Lines 1257-1258 and 1329-1332: The code comment claims a "9-order-of-magnitude power gap (4 MW cloud to 0.1 W TinyML)." But 4 MW / 0.1 W = 40,000,000 = 4e7, about 7.6 orders of magnitude, not 9. The same representative table lists ESP32-CAM power as 0.1-0.5 W, not mW-scale power.
  - Proposed correction: Either change the comment to "about 8 orders of magnitude for the representative ESP32-CAM row" or use a true 1 mW TinyML target consistently, where 4 MW / 1 mW = 4e9, about 9.6 orders of magnitude.

- Lines 1332 and 1345: The representative TinyML row lists ESP32-CAM power as 0.1-0.5 W, while the decision-threshold table says TinyML power is <1 mW. These cannot both describe the same platform/threshold without explanation: 0.1 W is 100 mW, already 100x above the <1 mW threshold.
  - Proposed correction: Split "TinyML always-on inference target: <1 mW" from "ESP32-CAM board power: 100-500 mW", or choose a sub-mW MCU/sensor example for the representative TinyML row.

### Medium severity

- Line 937: "six-order-of-magnitude range in compute (MW cloud vs. mW TinyML)" mixes compute with power units. MW and mW measure power, not compute. The same sentence also says cost spans six orders from "$millions vs. $10"; $1,000,000 / $10 = 100,000, which is five orders, not six unless the upper endpoint is at least $10,000,000.
  - Proposed correction: Change "compute" to "power" and state "five to six orders of magnitude in cost" or give explicit endpoints that support six orders.

- Lines 1195 and 1197: The cost-spectrum footnote says the range is six orders of magnitude but then calls it "100,000x", which is five orders of magnitude. The PUE footnote repeats "six-order-of-magnitude economic range."
  - Proposed correction: Use "five-order-of-magnitude" for a 100,000x range, or change the range to 1,000,000x with endpoints such as $10 to $10M.

- Lines 1344 and 1331: The deployment-threshold table says Mobile ML power is <2 W, while the representative hardware table and mobile section use a 2-5 W mobile thermal envelope. This makes the mobile decision threshold stricter than the chapter's own stated operating range.
  - Proposed correction: Change the threshold to "2-5 W sustained envelope" or clarify that "<2 W" is a continuous/background target distinct from peak or short-burst mobile ML.

- Lines 2014-2017 vs. 2059-2080: The bandwidth-bottleneck code comment says 100 1080p cameras produce about 5 GB/s and need 5x a 10 Gbps link. The actual formula uses 1920 x 1080 x 3 bytes x 30 FPS x 100 cameras, which is about 18.7 GB/s. A 10 Gbps link is about 1.25 GB/s, so the shortfall is about 15x.
  - Proposed correction: Update the comment to "about 18.7 GB/s" and "about 15x", matching the code and rendered calculation.

- Line 2132: The claim that sending only ~1 KB defect metadata reduces bandwidth by 1,000,000x is not supported by the preceding calculation without a detection-rate assumption. From the raw 100-camera stream of about 18.7 GB/s, one 1 KB metadata event per camera per second would be about 100 KB/s, a reduction of about 187,000x; one metadata event per frame would be about 3 MB/s, only about 6,200x.
  - Proposed correction: Add the assumed metadata event rate and recompute the reduction, or replace with a defensible range such as "thousands to hundreds of thousands of times."

- Lines 2252 and 2262: The locality example says "4K, 60 FPS" but the calculation uses only one 4K frame: 3840 x 2160 x 3 bytes ~= 24.9 MB, then 24.9 MB x 8 / 100 Mbps ~= 1991 ms. The 60 FPS stream rate is not used.
  - Proposed correction: Either say "one 4K frame from a 60 FPS stream" and compare per-frame latency, or include the stream rate: 24.9 MB/frame x 60 FPS ~= 1.49 GB/s, far above a 100 Mbps uplink.

- Line 2960: "Compared to cloud systems, TinyML deployments provide 10^4 to 10^5 times less memory" is too small for the chapter's own representative hardware. Using 131 TB cloud memory and 256 KB-1 MB TinyML memory gives roughly 1.3e8 to 5.1e8 difference.
  - Proposed correction: Change to "10^8 to 10^9 times less memory" for cloud vs. TinyML, or change the comparator to mobile if the intended range is around 10^4.

- Line 4198: The Amdahl's Law footnote says if ML inference is 30-50 percent of pipeline time, even a 100x model speedup yields at most 2-3x end-to-end improvement. For p = 0.30, speedup = 1/(0.70 + 0.30/100) ~= 1.42x; for p = 0.50, speedup = 1/(0.50 + 0.50/100) ~= 1.98x. The stated range should be about 1.4-2.0x, not 2-3x.
  - Proposed correction: Change to "at most about 1.4-2x" for the stated 30-50 percent range, or change the ML fraction to about 50-67 percent if retaining "2-3x."

### Low severity

- Lines 1175 and 1185: The ResNet callout says the bottleneck term is "`ratio`x slower than compute." This is correct if `calc_bottleneck()["ratio"]` is memory time divided by compute time, but the reportable formula is implicit.
  - Proposed correction: In the prose, define the displayed ratio as `T_mem / T_comp` to avoid ambiguity.

- Line 2231: "processing 1000 camera feeds locally avoids 1 Gbps uplink costs" appears understated relative to the earlier raw-video example: even 100 uncompressed 1080p RGB cameras produce far more than 1 Gbps. This may assume compressed camera feeds, but that assumption is not stated.
  - Proposed correction: Add "compressed" or provide a per-camera bitrate assumption, e.g. "1000 compressed camera feeds at 1 Mbps each avoids about 1 Gbps of uplink."

## Checked calculations with no issues found

- Lines 327-369: Light-speed round-trip latency equation is dimensionally correct; 2 x 3600 km / 200,000 km/s = 0.036 s = 36 ms.
- Lines 380-421: If voltage scales proportionally with frequency and capacitance is fixed for the comparison, dynamic power scales as V^2 f, so doubling frequency implies about 8x power.
- Lines 470-518: Memory-wall ratio 1.6 / 1.2 = 1.333..., matching 1.33x/year.
- Lines 529-538 and 945-947: Iron-law additive and pipelined max forms are dimensionally consistent.
- Lines 1594-1757: Cloud-vs-edge TCO arithmetic is internally consistent from the displayed assumptions.
- Lines 3296-3349: Braking-distance conversion is correct: 100 km/h = 27.78 m/s, and 100 ms corresponds to 2.8 m traveled.
- Lines 4082-4134: TCO pitfall arithmetic is correct: 500 + 3000 + 500 + 2000 = 6000, and 6000 / 2000 = 3x.
- Lines 4136-4196: Amdahl camera example arithmetic is correct: 100 + 60 + 40 = 200 ms; 60 ms / 10 = 6 ms; optimized total = 146 ms; speedup = 200 / 146 ~= 1.37x; theoretical infinite-ML speedup = 1 / 0.70 ~= 1.43x.
