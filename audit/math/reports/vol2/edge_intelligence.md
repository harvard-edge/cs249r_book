# Math Audit Report: `book/quarto/contents/vol2/edge_intelligence/edge_intelligence.qmd`

## Checked scope

Audited the edge intelligence chapter for on-device and federated-learning math, communication/latency/energy/memory calculations, numeric examples, unit conversions, scaling/statistical claims, and prose-equation consistency using direct reasoning only. No Gemini assistance was used. No source `.qmd` files were modified.

## Findings

### High Severity

- **Lines 625-627: the depthwise-separable convolution parameter calculation is off by about 19x.**  
  A standard `3 x 3` convolution with 512 input and 512 output channels has `3 x 3 x 512 x 512 = 2,359,296` weights, about `2.4M`, as stated. The depthwise-separable version has `3 x 3 x 512 = 4,608` depthwise weights plus `1 x 1 x 512 x 512 = 262,144` pointwise weights, for `266,752` weights, not `13.8K`. The reduction is about `2,359,296 / 266,752 = 8.8x`, not `174x`.
  - Proposed correction: Replace `13.8 K` and `174x` with approximately `267K` and `8.8x`. If the intended comparison uses a much smaller output-channel count, state that channel count explicitly and recompute the standard convolution accordingly.

- **Lines 2253-2273 and 2281-2295: the FedAvg convergence bound becomes zero in the IID case, contradicting the worked example.**  
  The displayed non-IID bound is `O(beta / sqrt(CER) + beta^2 E^2 / R)`. Line 2271 says IID data has `beta = 0` and the bound reduces to the standard rate, but substituting `beta = 0` makes both terms zero for any finite `C`, `E`, and `R`. The worked IID example then switches to a separate variance bound `epsilon <= sigma / sqrt(CER)`, which is the missing base term.
  - Proposed correction: Use a bound with a nonzero stochastic optimization term plus a heterogeneity penalty, for example `O(sigma / sqrt(CER) + beta^2 E^2 / R)`, or define `beta` so that it does not vanish in the IID stochastic term. Then recompute or relabel the worked examples under the same bound.

- **Line 2976: the sustained-power-to-energy conversion is wrong by 1000x.**  
  The text says `500-1000 mW` sustained power translates to `1.8-3.6 joules per hour`. But `0.5 W x 3600 s = 1800 J` and `1.0 W x 3600 s = 3600 J`. The stated `1.8-3.6 J` would correspond to only `0.5-1.0 mW` for one hour.
  - Proposed correction: Change this to `1.8-3.6 kJ per hour`, or equivalently `0.5-1.0 Wh per hour`.

- **Line 2992: the microcontroller battery-life example is dimensionally wrong by roughly two orders of magnitude.**  
  The footnote says a device drawing `10 mW` exhausts a `1000 mAh` battery in `2.8 hours`. A `1000 mAh` battery stores `1 Wh` at `1 V`, `3.7 Wh` at a typical Li-ion voltage, and around `3 Wh` at `3 V`. At `10 mW`, runtime is therefore about `100-370 hours`, not `2.8 hours`. The same footnote says training for `10 seconds every hour` consumes about `1 joule`; at `10 mW`, `10 s` consumes `0.01 W x 10 s = 0.1 J` per hourly training event.
  - Proposed correction: Either change the training power to about `1.3 W` if the intended runtime is `2.8 h` on a 3.7 Wh battery, or change the runtime to about `100-370 h` for a 10 mW load. Change the duty-cycled event energy to `0.1 J per 10-second training burst` at 10 mW.

### Medium Severity

- **Lines 667 and 952: the Apple Neural Engine improvement factor conflicts with the chapter's own TOPS value.**  
  Line 667 says A11 `0.6 TOPS` to A17 Pro `{python} EdgeIntelligenceSetup.mobile_npu_tops` TOPS is a `58x` improvement. The setup comment and later prose use `16 TOPS`, so the factor is `16 / 0.6 = 26.7x`, not `58x`.
  - Proposed correction: Change `58x` to about `27x`, or use an A17 figure of about `35 TOPS` if `58x` is intended.

- **Lines 1399-1401: the TinyTL memory-savings numbers mix incompatible ratios and understate full-training memory.**  
  Reducing trainable parameters from `3.4M` to `50K` is a `68x` reduction, not `60x` unless rounded very coarsely from `3.0M / 50K`. The footnote's memory collapse from `20 MB` to `600 KB` is a `33x` reduction, while the prose says TinyTL reduces training memory by `10x`. Also, MobileNetV2 FP32 weights alone are `3.4M x 4 = 13.6 MB`; full training memory including gradients and activations is unlikely to be only `20 MB` under the chapter's own amplification discussion.
  - Proposed correction: Use one consistent parameter baseline and one consistent memory baseline. For example: `3.4M to 50K is about 68x fewer trainable parameters`; then give a separate measured/assumed training-memory reduction, with components listed.

- **Line 1433: the rank-4 low-rank example reports the wrong percent reduction.**  
  A `768 x 768` matrix has `589,824` parameters. A rank-4 update has `768 x 4 x 2 = 6,144` parameters, which is `1.04%` of the full matrix. That is a `98.96%` reduction, not `96%`. A `96%` reduction would correspond to roughly rank 16 for this square matrix.
  - Proposed correction: Change `96 percent reduction` to about `99 percent reduction`, or change the example to rank 16 where `24,576 / 589,824 = 4.17%`, a `95.8%` reduction.

- **Lines 1442-1446: LoRA memory and transfer examples conflate model storage, full fine-tuning state, and network throughput.**  
  A 7B FP16 model has about `14 GB` of weights, but full fine-tuning also requires gradients and optimizer state, commonly several times larger than the weight file. Calling `14 GB` the requirement for "full fine-tuning" understates training memory. The same section says `10-50 MB` LoRA adapters can synchronize in under `30 s` on 3G; `50 MB = 400 Mb`, so even an optimistic `2 Mbps` 3G uplink takes about `200 s`, and `10 MB` takes about `40 s`.
  - Proposed correction: Say `14 GB` is the FP16 weight storage or full-model transfer size, not full fine-tuning memory. Change the 3G synchronization claim to minutes for `10-50 MB`, or use a faster 4G/5G uplink for the under-30-second claim.

- **Lines 2420-2425: the second FedAvg formula uses stale round notation for the local model.**  
  The prose says `theta_k^t` represents the locally updated model parameters, and the formula averages `theta_k^t` to produce `theta^{t+1}`. Earlier, lines 2091-2095 use the clearer convention that clients start from round `t`, perform local training, and produce `theta_k^{t+1}`, which the server averages into `theta^{t+1}`. The second version can be read as averaging pre-update models.
  - Proposed correction: Change the local updated model in line 2420 and the formula on line 2422 to `theta_k^{t+1}` or define `theta_k^t` as "the post-local-training model submitted during round t" explicitly.

- **Lines 2429-2520: the communication-computation figure code does not model the drift mechanism described in the caption.**  
  The plotted curves are `A/E + E + 5` scaled by constants. The right-side increase comes from the linear local-compute term `E`, not from model drift requiring more rounds to converge. The caption says excessive local computation increases total time due to drift, which would require a convergence-round multiplier or heterogeneity penalty that grows with `E`.
  - Proposed correction: Either relabel the plotted model as a simple communication-plus-local-compute cost curve, or add a drift/convergence penalty term such as `gamma E^2` or a rounds-to-convergence multiplier for the non-IID case.

- **Lines 2524-2562: the Top-k compression example omits index overhead and therefore overstates the 8x saving.**  
  The standard four FP32 values are `4 x 32 = 128 bits`; four INT8 values are `32 bits`, so the `4x` quantization label is correct. For Top-k, transmitting two 8-bit values is `16 bits`, but sparse transmission also needs positions. Even in a length-4 toy vector, two indices require at least `4` more bits; in real vectors, the index cost is about `k log2 N` bits before coding. If values remain FP32, the payload is at least `64 bits` plus indices.
  - Proposed correction: Label the toy example as "values only" or include index overhead. For a general Top-k update, state payload as approximately `k(value_bits + log2 N)` bits plus metadata/residual handling.

- **Line 2406: the LTE upload energy estimate is likely too high for the stated transfer time.**  
  A `50 MB` upload is `400 Mb`, so the `40-80 s` at `5-10 Mbps` is correct. But `100 mAh` at a phone battery voltage around `3.7 V` is about `0.37 Wh` or `1.3 kJ`. Sustaining even `2 W` radio/system power for `80 s` consumes `160 J`, about `12 mAh` at `3.7 V`; `100 mAh` would imply an average power near `17 W` for 80 seconds, which is not consistent with the surrounding phone-power envelope.
  - Proposed correction: Recompute the energy from an explicit radio/system power assumption, e.g. `1-3 W x 40-80 s = 40-240 J`, corresponding to roughly `3-18 mAh` at `3.7 V` and about `0.1-0.4%` of a 15 Wh battery.

- **Line 2681: the device-heterogeneity ratios are inconsistent with the cited endpoints.**  
  The text says compute variation spans `1000x` between a flagship phone at `{python} EdgeIntelligenceSetup.mobile_npu_tops` TOPS and an IoT microcontroller at `0.03 TOPS`. With the chapter's `16 TOPS` value, the ratio is `16 / 0.03 ≈ 533x`; `1000x` would require about `30 TOPS`. The same sentence says memory differences are `100-10,000x`, but `16 GB / 256 KB` is about `62,500x`, which is larger than the stated range.
  - Proposed correction: Change compute variation to about `500x` for the stated endpoints, or update the flagship TOPS. Change the memory range to include about `60,000x`, or use less extreme memory endpoints.

- **Lines 2787-2789: the brain-efficiency comparison is not supported by the stated operation and power numbers.**  
  If the brain performs `10^15` operations/s at `20 W`, that is `5 x 10^13 operations/J`. A modern AI accelerator at roughly `10^15 operations/s` and hundreds of watts is on the order of `10^12 operations/J`, making the ratio tens of times, not `50,000x`, under this operation-count comparison. The `50,000x` claim may come from a different definition of operation or from comparing to much less efficient hardware, but it does not follow from the numbers shown.
  - Proposed correction: Either remove the `50,000x` figure, qualify the comparator precisely, or recompute using a specified accelerator energy per operation and a compatible definition of "operation."

### Low Severity

- **Lines 333-344, 430, 964, 3082, and 3111: the chapter uses several different training-memory amplification ranges without explaining the scope changes.**  
  The chapter alternates among `4-12x`, `3-5x`, and `3-10x` memory amplification. Some ranges may refer to full backpropagation, some to activation caching, and some to broader resource needs, but the prose often treats them as the same quantity.
  - Proposed correction: Define one primary memory-amplification range for full training relative to inference, and reserve narrower ranges for specific components such as activation caching or optimized adaptation.

- **Lines 196 and 430: several memory examples are directionally correct but should state which endpoint or component they use.**  
  `50-100 MB` against a `200-300 MB` keyboard-app budget can be anywhere from `17%` to `50%`; the stated `25%` follows from `50/200` or `75/300`. A 10M-parameter FP32 model has `40 MB` of weights; the figure code's default training stack is `9x` weights (`360 MB`), while line 430 says it spikes to "over 200 MB." Both can be true, but they rely on different activation assumptions.
  - Proposed correction: State the chosen denominator for the `25%` claim and either align the 10M model memory prose with the figure's `9x` stack or explicitly say that `200 MB` is a lower-bound/mobile-optimized estimate.

- **Lines 671 and 940: quantization and mixed-precision memory reductions are stated as broad constants without specifying which tensors are reduced.**  
  INT8 gives a `4x` reduction for FP32 weights or activations, but training memory includes gradients, optimizer state, FP32 master weights, and sometimes activation caches. Similarly, "INT8 for inference and FP16 for gradients" does not automatically yield `4x` lower total training memory than full FP32 if optimizer state remains FP32.
  - Proposed correction: Qualify these reductions as applying to the tensors stored at lower precision, or provide a component-wise memory budget for weights, gradients, activations, and optimizer state.

- **Lines 2655-2661: compression-ratio prose should distinguish precision compression from total update compression.**  
  FP32-to-INT8 quantization gives `4x` by itself, while `100-1000x` total update reduction requires sparsity, selective layers, entropy coding, low-rank/adapters, or not sending most parameters. The text mostly implies this, but line 2655 moves quickly from `10-100x` to `100-1000x`.
  - Proposed correction: Add one example decomposition, such as `4x from INT8 times 25x from top-4% sparsity = about 100x before metadata`.

- **Line 3094: the "6+ orders of magnitude" hardware-span claim is slightly overstated for the examples given.**  
  `32 KB` to `16 GB` is about `5 x 10^5`, `10 uW` to `5 W` is also about `5 x 10^5`, and `10 MIPS` to `100,000 MIPS` is `10^4`. These are enormous differences, but the listed endpoints are closer to four to under-six orders, not clearly `6+`.
  - Proposed correction: Change to "4-6 orders of magnitude" or use endpoints that exceed `10^6x`.

## Verified Correct

- **Lines 140-166:** The NPU speedup and energy-efficiency notebook is arithmetically correct: `400 ms / 20 ms = 20x`, and `0.8 J / 0.016 J = 50x`.
- **Lines 196:** The keyboard-memory example is plausible for the stated range: `50 MB / 200 MB = 25%`, though other endpoints produce different percentages.
- **Lines 228 and 2397:** The chapter consistently treats smaller `epsilon`/`varepsilon` as stronger differential privacy and uses plausible privacy-budget examples; no arithmetic contradiction was found there.
- **Lines 613-615:** The Arduino memory comparison is approximately correct under binary units: `16 GiB / 256 KiB = 65,536`, and a `224 x 224 x 3` 8-bit image is about `150 KB`, around `60%` of `256 KB`.
- **Lines 883-926:** The phone battery-drain notebook is correct: `4.5 W x 0.5 h = 2.25 Wh`, and `2.25 Wh / 15 Wh = 15%`.
- **Lines 1004-1033 and 1484-1486:** The adapter storage example is internally consistent: `10M x 4 bytes = 40 MB`, `50,000 x 4 bytes = 200 KB`, and `40 MB / 200 KB = 200x`.
- **Lines 1413-1431:** The residual-adapter and low-rank parameter formulas are algebraically correct: a bottleneck adapter has `rd + dr` matrix parameters, and a rank-`r` update for an `m x n` matrix has `r(m+n)` parameters.
- **Lines 1728-1730:** The MFCC compression example is arithmetically sound: a 20 ms, 16 kHz, 16-bit audio window is `320` samples or `640 bytes`; `12-13` FP32 coefficients are `48-52 bytes`, about a `12-13x` reduction.
- **Lines 1780-1827:** The federated communication savings notebook is correct under decimal MB for updates and binary conversion for raw images: `1000 x 200 KB / 1024 = 195 MB`, `5M x 4 bits / 8 = 2.5 MB`, and `195 / 2.5 = 78x`.
- **Lines 2091-2095:** The first FedAvg weighted-average formula is consistent when `theta_k^{t+1}` denotes each client's post-local-training model and `n = sum_k n_k` over active clients.
- **Lines 2281-2303:** Given the chapter's worked-example formulas, the arithmetic is correct: IID requires `200` rounds, non-IID with `E=5` gives `5,625` rounds, and non-IID with `E=2` gives `900` rounds.
- **Lines 2406 and 2657:** The transfer-time conversions are correct: `50 MB = 400 Mb`, so `5-10 Mbps` takes `40-80 s`; `8 MB = 64 Mb`, so `5-50 Mbps` takes about `1.3-13 s`.
- **Lines 2799-2801:** The sensor-rate examples are arithmetically correct: `30 FPS x 86,400 s/day = 2.592M frames/day`, and `100 Hz x 86,400 s/day = 8.64M accelerometer samples/day`.
