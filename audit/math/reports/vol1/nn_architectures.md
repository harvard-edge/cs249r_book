# Math Audit Report: `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd`

## Checked scope

Audited equations, derivations, numeric examples, unit conversions, complexity/scaling claims, tensor-shape claims, and prose-equation consistency using direct reasoning only. No source `.qmd` files were modified.

## Findings

### High Severity

- **Line 4711: KV-cache memory calculation omits either keys or values.**  
  The prose describes a key-value cache, but the displayed calculation uses `32 layers x 32 heads x 2048 tokens x 128 head_dim x 2 bytes`, which counts only one tensor family. A KV cache stores both K and V, so the per-request FP16 cache is `32 x 2 x 32 x 2048 x 128 x 2 = 1,073,741,824` bytes, about **1.07 GB** decimal, not 537 MB. The later statement that 2--4 users consume 1--2 GB is based on the omitted-factor result; with both K and V, 2--4 users consume about **2.15--4.29 GB**.
  - Proposed correction: Insert the missing `x 2 (K,V)` factor and update the concurrency totals. This also aligns with the correct formula already used in the KV-cache footnote at line 3053.

### Medium Severity

- **Lines 371 and 344-356: "three-order-of-magnitude gap" conflicts with the displayed arithmetic.**  
  The code gives ResNet intensity `4.1e9 / 102e6 = 40.2` FLOPs/byte and GPT-2 intensity `3.0e9 / 6.0e9 = 0.5` FLOPs/byte. The ratio is about **80x**, less than two orders of magnitude, not three orders. MobileNet is `300e6 / 14e6 = 21.4` FLOPs/byte.
  - Proposed correction: Change "three-order-of-magnitude gap" to "roughly two-order-of-magnitude gap" or "about 80x gap" for the current constants.

- **Lines 1515, 1522, and 1524: translation-equivariance derivation contains invalid variable names.**  
  The derivation defines the translation vector as `v = (v_1, v_2)` on line 1513, but then uses `v_one` in three mathematical expressions. That makes the notation inconsistent and breaks the derivation as written.
  - Proposed correction: Replace every `v_one` in the derivation with `v_1`.

- **Lines 3475-3480: scalar residual-gradient product overstates the guarantee.**  
  The multi-block expression treats each residual derivative as a scalar factor `F'_l(x_l) + 1` and then says each factor has expectation at least one if `F'_l` is non-negative on average. In actual networks the factors are Jacobians `I + J_F`, and residual-branch eigenvalues can be negative even when the identity path improves conditioning. The identity shortcut creates an additive gradient route, but it does not guarantee every multiplicative factor maintains or increases gradient magnitude.
  - Proposed correction: Keep the one-block identity-path explanation, but soften the multi-block claim: "residual blocks add an identity component to each Jacobian, tending to keep eigenvalues closer to one when the residual branch is small," consistent with lines 3492-3497.

- **Lines 3582 and 3575-3579: BatchNorm identity parameters ignore epsilon and batch-dependent statistics.**  
  From `y_i = gamma * (x_i - mu_B) / sqrt(sigma_B^2 + epsilon) + beta`, exact identity for fixed batch statistics would require `gamma = sqrt(sigma_B^2 + epsilon)` and `beta = mu_B`, not simply `gamma = sigma_B`. More importantly, learned `gamma` and `beta` are fixed parameters, while `mu_B` and `sigma_B` change by batch during training, so the statement should not imply the layer can exactly recover identity for every mini-batch by setting learned constants to current batch statistics.
  - Proposed correction: Say BatchNorm preserves representational capacity approximately/under fixed statistics, with `gamma` matching `sqrt(sigma_B^2 + epsilon)` and `beta = mu_B`.

- **Lines 3621 and 3623-3631: normalization memory comparison mislabels state as parameters and omits LayerNorm/RMSNorm learned parameters.**  
  BatchNorm's running mean and variance are persistent buffers, not learned parameters. BatchNorm, LayerNorm, and RMSNorm all commonly have learned scale/shift parameters except RMSNorm often omits bias in some implementations. Saying BatchNorm has `2 x H additional parameters` while LayerNorm has "no persistent memory overhead" is inconsistent because LayerNorm also stores at least `gamma` and usually `beta`.
  - Proposed correction: Distinguish learned affine parameters from running-stat buffers. For example: "BatchNorm stores learned gamma/beta plus running mean/variance buffers; LayerNorm stores learned gamma/beta but no running statistics; RMSNorm stores learned gamma and often no beta."

- **Lines 3357-3359: the capacity-wall conclusion should say single GPU, not single machine.**  
  The arithmetic for one table is `100M x 128 x 4 = 51.2 GB`. Two same-sized tables are about **102.4 GB**, which exceeds an 80 GB A100-class GPU but does not generally exceed a "single machine" with CPU RAM or multiple GPUs. The preceding comparison is explicitly against one A100 GPU.
  - Proposed correction: Change "cannot fit on a single machine" to "cannot fit on a single 80 GB GPU" or qualify the machine configuration.

### Low Severity

- **Line 66: the 224x224 CNN-vs-MLP parameter reduction is understated and loosely specified.**  
  For one feature detector over a 224 x 224 RGB image, an MLP-style dense detector has `224 x 224 x 3 = 150,528` weights, while a 3 x 3 RGB CNN filter has `3 x 3 x 3 = 27` weights. The ratio is about **5,575x**, not roughly 1,000x. "Roughly 1,000x" is directionally correct as an order-of-magnitude claim, but the chapter later computes the exact 5,576x-style value at lines 1435-1454 and 1602.
  - Proposed correction: Either use "over 5,000x" here or state that the exact ratio depends on image and kernel size.

- **Lines 903-923: dense-layer example mixes row-vector multiplication with column-vector output display.**  
  The text explicitly treats `h^(0)` as a row vector later, and `h^(0) W^(1)` with a `4 x 3` matrix produces a `1 x 3` row vector. The numeric entries `[0.59, -0.09, 0.45]` are correct, but the computation is displayed as a column vector.
  - Proposed correction: Display the result as a row vector or say the column display is a transposed presentation for readability.

- **Lines 1964-1968 and 2108-2118: RNN weight notation flips between `W_hx` and `W_xh`.**  
  The equation uses `W_hx x_t`, while the figure caption and implementation use `W_xh` for input-to-hidden weights. Both conventions can be valid, but the chapter presents them as the same object.
  - Proposed correction: Standardize on one name, likely `W_xh` for input-to-hidden, and update the equation/prose accordingly. If using column-vector convention in the equation, also specify the expected shapes.

- **Line 3648: LSTM gate count and cost explanation is internally inconsistent.**  
  The footnote says three gates triple the parameter count and GEMM operations, then says each LSTM step is about 4x more expensive. A standard LSTM has three gates plus a candidate update, i.e. four affine transformations compared with one in a vanilla RNN, so the ~4x statement is the consistent one.
  - Proposed correction: Change "three gates per cell ... triple" to "three gates plus a candidate update ... roughly quadruple."

## Verified Correct

- Lines 344-356: arithmetic-intensity values from the constants are internally computed correctly: ResNet about 40.2 FLOPs/byte, GPT-2 about 0.5 FLOPs/byte, MobileNet about 21.4 FLOPs/byte.
- Lines 425-450 and 511-515: lighthouse parameter, FLOP, memory, and MobileNet-vs-ResNet ratio calculations are structurally consistent with the constants used in code.
- Lines 669-696 and 708-725: MNIST MLP/CNN parameter arithmetic is correct: MLP about 20.0M parameters; CNN about 421K parameters; integer reduction ratio 47x.
- Lines 914-923: the dense-layer numeric multiplication is arithmetically correct: `[0.8, 0.2, 0.9, 0.1] W = [0.59, -0.09, 0.45]`, followed by ReLU `[0.59, 0, 0.45]`.
- Lines 1122-1138: a 2048 x 2048 dense layer has 4,194,304 parameters and about 16 MB at FP32.
- Lines 1435-1454 and 1602: the CNN parameter-sharing calculation is correct: `224 x 224 x 3 = 150,528`, `3 x 3 x 3 = 27`, ratio about 5,575x.
- Lines 1576-1595: the concrete shifted-edge convolution output is consistent with a valid 3 x 3 convolution and shifts the response by two columns.
- Lines 1860-1898: CNN system calculations are correct for the stated assumptions: 576 weights for `3 x 3 x 1 x 64`, 50,176 spatial positions, and 3,211,264 activation values for `224 x 224 x 64`.
- Lines 1912-1917: depthwise-separable convolution reduction is correct in ratio form: `(K^2 C_in + C_in C_out)/(K^2 C_in C_out) = 1/C_out + 1/K^2`.
- Lines 2086-2110: RNN per-step MAC counts are correct for 100 input and 128 hidden dimensions: 16,384 recurrent + 12,800 input = 29,184 MACs.
- Lines 2195 and 2701-2744: attention matrix memory calculations are correct under their stated assumptions: `4096^2 x 2` bytes is about 32 MiB/33.6 MB, and `100,000^2 x 12 heads x 2 bytes = 240 GB` per layer.
- Lines 2412 and 2541-2556: QKV projection shape and MAC count are correct: `(6 x 768)(768 x 2304) -> (6 x 2304)` with 10,616,832 MACs.
- Lines 2573-2610 and 2754: attention score/value aggregation MAC counts are correct for one head at `N=512`, `d=64`: `512 x 512 x 64 = 16.8M` MACs for scores and the same order for value aggregation.
- Lines 2803-2822: scaled dot-product self-attention and multi-head attention formulas are dimensionally consistent when the usual projection shapes are assumed.
- Lines 3237-3255 and 3330-3359: embedding table arithmetic is correct: `1B x 128 x 4 = 512 GB`, and `100M x 128 x 4 = 51.2 GB`.
- Lines 4450-4485: throughput-ceiling arithmetic is correct: ResNet-50 at 30 FPS requires about 123 GFLOPs/s; a 5 TFLOP/s effective GPU gives about 41x headroom; a 100 GFLOP/frame detector at 30 FPS requires 3 TFLOP/s and leaves about 2x headroom.
- Lines 4514-4556 and 4607-4633: wildlife sizing arithmetic is correct: MobileNetV2 0.75x has 2.2M params, about 8.8 MB FP32 and 2.2 MB INT8; 200 mW for 75 ms is 15 mJ; 100/day is 1.5 J/day.
- Lines 4621-4625: the worked memory and latency checks are arithmetically correct after rounding: `224 x 224 x 64 x 4 = 12.85 MB`, so `2.2 + ~12.85 + 50 ~= 65 MB`; `150 MOPs / 2 GOPS = 75 ms`.
