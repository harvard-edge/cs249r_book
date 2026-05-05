# Math Audit: vol1/nn_computation/nn_computation.qmd

Source audited: `book/quarto/contents/vol1/nn_computation/nn_computation.qmd`

## Summary

The chapter's core neural-network algebra is mostly coherent under its row-vector convention: layer shapes, parameter counts for the canonical `784 -> 128 -> 64 -> 10` MLP, cross-entropy formulas, and the worked backpropagation arithmetic are largely correct. The main issues are in systems-facing quantitative claims: training-vs-inference memory mixes batch sizes, backpropagation complexity is described as `O(1)`, some scaling ratios are inconsistent with the code, and one figure/table uses inconsistent dimensions or units.

No source `.qmd` files were modified.

## Findings

### 1. Training-vs-inference memory comparison mixes batch sizes

- Lines 3041-3049 compute training memory with batch size 32 but inference memory with batch size 1.
- Lines 3327-3333 summarize inference activations using the batch-32 activation memory, while the inference total uses batch-1 activation memory.
- Line 3337 says training requires `MNISTMemory.training_ratio_str` times more memory "for the same batch size."
- Severity: High.
- Issue: The reported ratio is not for the same batch size. Direct arithmetic for the `784 -> 128 -> 64 -> 10` MLP gives 109,386 parameters, or 427.3 KiB of FP32 parameter memory. Batch-32 activations over input plus all layer outputs are `(32*(784+128+64+10)*4)/1024 = 123.25 KiB`. Training memory with parameters, gradients, Adam's two states, and batch-32 activations is `427.3 + 427.3 + 854.6 + 123.25 = 1832.4 KiB`. Inference with batch size 1 is `427.3 + ((784+128+64+10)*4)/1024 = 431.1 KiB`, giving the shown ~4.3x ratio. Inference with batch size 32 is `427.3 + 123.25 = 550.5 KiB`, giving only ~3.3x.
- Proposed correction: Either state explicitly that the comparison is "training at batch size 32 vs single-sample inference" and change line 3330 to the batch-1 activation memory, or recompute `inference_total_kb` for batch size 32 and change the ratio to ~3.3x.

### 2. Backpropagation is not `O(1)` in computational complexity

- Lines 3979-3980: backpropagation is said to compute gradients in `O(1)` backward passes and to scale independently of the number of parameters.
- Severity: High.
- Issue: Backpropagation needs one reverse traversal of the computation graph, but its work and memory scale with the graph size and parameter count. For a dense layer, computing `dW = A^T dZ` has the same asymptotic matrix-multiply scale as the corresponding forward layer. The correct contrast with numerical differentiation is one backward pass through the graph versus `P` forward passes for `P` parameters, not `O(1)` total computation.
- Proposed correction: Replace with: `It computes all parameter gradients in one backward traversal of the computation graph, with cost on the same order as a small constant multiple of the forward pass, instead of O(P) forward passes for numerical differentiation over P parameters.`

### 3. Training memory equation omits gradient storage

- Lines 4007-4008 define `Training Memory ≈ Model Weights + Optimizer States + Activations`.
- Lines 4003 and 4113 correctly discuss gradient storage.
- Severity: Medium.
- Issue: The equation omits gradients even though gradients are a required training-time tensor and are included in the chapter's own memory calculations. With Adam, the usual training footprint is weights + gradients + two optimizer-state tensors + activations, before framework overhead and temporary buffers.
- Proposed correction: Change the displayed equation to: `Training Memory ≈ Model Weights + Gradients + Optimizer States + Activations`.

### 4. Operation-count ratio is documented as ~1,500x but computes to ~1,092x

- Lines 385-386 in the `paradigm-systems-cost` cell comment say the shift is a "~1,500x MAC increase" from rule-based to deep learning.
- Lines 412-444 compute `rb_ops = 100`, `dl_total_macs = 109,184`, and `dl_ops_ratio = 1,091.84`, formatted as `1,092`.
- Lines 724, 766, 5254, and 5260 use the computed `1,092x` value.
- Severity: Low.
- Issue: The rendered prose is consistent with the computation, but the executable-cell documentation is not. The direct ratio is `109,184 / 100 = 1,091.84`, not 1,500.
- Proposed correction: Update the comment to "~1,100x" or "~1,092x".

### 5. GPT-2 memory-explosion comments conflict with the actual constants and rendered values

- Lines 3354-3356 describe GPT-2 as 124M parameters and ~475 MB.
- Lines 3393-3406 compute from `GPT2_PARAMS`; the comments on lines 3403-3406 expect about 1.558B parameters, 6 GB, and a 14,244x jump.
- Lines 3413-3416 render the computed values into the prose.
- Severity: Medium.
- Issue: The source cell's stated goal describes GPT-2 small (124M parameters), but the code and expected output use GPT-2 XL scale (~1.558B parameters). The memory ratio differs by more than an order of magnitude: 124M / 109,386 ≈ 1,134x and FP32 memory is about 496 MB decimal, while 1.558B / 109,386 ≈ 14,244x and FP32 memory is about 6.2 GB decimal.
- Proposed correction: Either change the goal/comment and prose label to "GPT-2 XL", or set the constant/example to GPT-2 small and update the memory and jump values to ~496 MB and ~1,134x.

### 6. HOG cell-size prose conflicts with the MNIST worked example

- Line 580 says HOG computes orientations in fixed `8 x 8` pixel cells.
- Lines 421-430 and 590 use a `28 x 28` MNIST image divided into a `7 x 7` grid of `4 x 4` cells with 9 bins, producing 441 features.
- Severity: Low.
- Issue: The worked example is internally correct: `28/4 = 7`, and `7*7*9 = 441`. But the nearby footnote describes HOG using `8 x 8` cells without qualifying that this is the common Dalal-Triggs setting rather than the chapter's MNIST toy setting. If a reader applies `8 x 8` cells to a `28 x 28` image, the `7 x 7` grid and 441-feature count do not follow.
- Proposed correction: Add a qualifier such as: `Classical HOG often uses 8 x 8 cells; the MNIST toy calculation below uses 4 x 4 cells so 28 x 28 divides into a 7 x 7 grid.`

### 7. Matrix-multiplication intensity table omits bytes in the IO term

- Lines 3691-3695 list matrix multiplication as `2N^3` ops, `3N^2` data movement, and intensity `≈ 2N/3` FLOPs/byte.
- Severity: Medium.
- Issue: The intensity expression is dimensionally wrong if the IO column counts elements. For FP32, moving three `N x N` matrices is `3N^2 * 4` bytes, so the naive intensity is `(2N^3)/(12N^2) = N/6` FLOPs/byte. The table's `2N/3` is FLOPs per element moved, not FLOPs per byte. If the table intends bytes, it needs a bytes-per-element factor.
- Proposed correction: Change the IO entry to `3N^2 elements (= 12N^2 bytes for FP32)` and the intensity to `N/6` FLOPs/byte for FP32, or relabel the intensity column as FLOPs/element.

### 8. Figure data for the MNIST image panel uses 29 columns while labels say 28 px

- Lines 2802-2822 comment that the data represent `28 x 28 = 784` pixel values.
- Lines 2827-2828 set the rendered raster width to `29` and height to `20`.
- Lines 2843-2844 label the panel as `28 px` by `28 px`.
- Severity: Low.
- Issue: The figure code is a visual excerpt, but the drawing dimensions are not consistent with the label. The code lays out data with 29 columns and 20 rows, while the figure annotation says 28 by 28. This is likely harmless pedagogically, but it is a tensor-shape/prose mismatch.
- Proposed correction: Either use a true 28 by 28 rendered grid or label the rendered raster as an illustrative excerpt while keeping the prose's `28 x 28 = 784` statement for the actual MNIST input.

### 9. Batch averaging prose overstates learning-rate invariance

- Lines 3900-3902 define average batch loss and say averaging makes the loss independent of batch size "so the same learning rate works whether B = 32 or B = 256."
- Line 4170 later notes that learning rate often couples to batch size.
- Severity: Low.
- Issue: Averaging does keep the gradient scale roughly comparable across batch sizes, but it does not imply the same learning rate will work equally well when changing batch size. Larger batches change gradient noise and often permit or require learning-rate adjustment.
- Proposed correction: Replace the learning-rate clause with: `so gradient magnitudes are on a comparable scale across batch sizes, although the optimal learning rate may still change with batch size.`

### 10. Backpropagation memory example says "several megabytes" for about 2 MiB

- Lines 4106-4113 discuss the wider `784 -> 512 -> 256 -> 10` network and say gradient storage requires several megabytes.
- Lines 3104-3125 compute 535,818 parameters for this wider network.
- Severity: Low.
- Issue: Direct arithmetic gives `535,818 * 4 / 1024 = 2,093 KiB`, about 2.0 MiB. "Several megabytes" is an overstatement unless including parameters, gradients, and optimizer states together. The full parameter+gradient+Adam-state memory is about `4 * 2.0 MiB = 8.2 MiB`, but gradient storage alone is not several MiB.
- Proposed correction: Say `gradient storage requires about 2 MiB` or `parameters, gradients, and Adam state together require about 8 MiB`.

## Checked But No Issue

- Lines 87-103 and 3167: Canonical MLP parameter count is correct: weights are 100,352, 8,192, and 640; biases are 128, 64, and 10; total parameters are 109,386.
- Lines 94-97, 117, 724, and 3657: Canonical forward MAC count is correct: `784*128 + 128*64 + 64*10 = 109,184`, and the first layer is nearly 100,000 MACs per sample.
- Lines 360 and 2215: Sigmoid vanishing-gradient examples are numerically correct: `0.25^20 ≈ 9.1e-13` and `0.25^10 ≈ 9.5e-7`.
- Lines 2207, 2223, 2237, 2255, 3739, 3919-3921, 3938, and 3949: Sigmoid, tanh, ReLU, softmax, cross-entropy, batch cross-entropy, and stable softmax formulas are mathematically standard.
- Lines 2618-2636, 3663-3670, and 3718-3723: The row-vector layer shapes are consistent: `A^(l-1)` has shape `B x n_{l-1}`, `W^(l)` has shape `n_{l-1} x n_l`, and the output has shape `B x n_l`.
- Lines 4069-4089: The worked backpropagation example is arithmetically correct: hidden pre-activation `[0.9, -0.2]`, output `0.54`, loss `0.1058`, output gradient `-0.46`, output weight gradient `[-0.414, 0]`, hidden signal `[-0.276, 0.184]`, ReLU-masked hidden gradient `[-0.276, 0]`, and updated `W^{(1)}_{11} = 0.5276`.
- Lines 4203-4205: MNIST epoch iteration count is correct for 60,000 examples and batch size 32: `60,000 / 32 = 1,875`.
- Lines 5123-5133: The improvement factors in the "Then vs. Now" table are internally consistent: `$50,000/$50 = 1,000`, `100 ms / 0.1 ms = 1,000`, `3 days / 30 seconds = 8,640`, and `10 J / 0.5 mJ = 20,000`.
