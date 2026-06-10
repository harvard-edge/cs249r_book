# Float-explanation worklist — `model_compression.qmd` (listings, pilot)

> Flag-only. Verdict scale: ✅ explained · ⚠️ announcer (points but content thin locally) · 🛑 bare-pointer / orphan.
> **Rubric note (calibrated on this chapter):** judge the reference in the context of its *paragraph + any post-float payoff prose*, NOT the ref sentence in isolation. An "X demonstrates Y" sentence is fine when the surrounding prose says what to notice; it is a finding only when no nearby prose explains the float.

## Listings (6)

### `lst-pruning_example` — line 577
- **Caption:** Magnitude-Based Pruning: removes weights below a threshold, nonzero 9 → 4 (k=4).
- **Ref (L573):** "…demonstrates this approach, removing weights with small absolute values to transform a dense weight matrix into the sparse representation visualized in @fig-sparse-matrix."
- **Verdict:** ✅ Explained — ref leads with the mechanism and links to the companion figure; caption gives the concrete count.
- **Fix:** none.

### `lst-quantization_example` — line 4012
- **Caption:** Uniform Quantization: FP32 → INT8, 4× memory reduction, measures error.
- **Ref (L4006):** "…demonstrates uniform quantization from FP32 to INT8, achieving 4× memory reduction while measuring the resulting quantization error."
- **Verdict:** ✅ Explained — ref carries the substantive takeaway (what + the 4× payoff + that it measures error).
- **Fix:** none.

### `lst-qat-conv-forward` — line 4878
- **Caption:** QAT Convolution Forward Pass: fake-quant nodes simulate INT8 while preserving gradient flow via STE.
- **Ref (L4876):** "…demonstrates the computational graph for a quantized convolution layer, which contains fake quantization nodes for both weights and activations."
- **Verdict:** ✅ Explained — ref says what the graph contains; post-listing prose (L4899+) walks the gradient-handling/STE payoff.
- **Fix:** none.

### `lst-conv-bn-relu-fusion` — line 5126
- **Caption:** Conv-BN-ReLU Fusion: three ops → one kernel, 6 → 2 transfers.
- **Ref (L5124):** "This ubiquitous Conv-BN-ReLU fusion pattern, illustrated in @lst-conv-bn-relu-fusion, appears in nearly every modern CNN architecture and reduces three memory round-trips to a single kernel launch."
- **Verdict:** ✅ Explained — **model construction**: content leads, ref rides along ("illustrated in"); the takeaway (round-trips collapsed) is in the sentence. Use as the positive template.
- **Fix:** none.

### `lst-qat_example` — line 7365
- **Caption:** Quantization-Aware Training: prepares a model to train in lower precision, accounting for quant error.
- **Ref (L7363):** "@Lst-qat_example demonstrates this pattern with PyTorch's `torch.ao.quantization` API."
- **Verdict:** ⚠️ Announcer (mild) — the ref sentence itself is thin ("demonstrates this pattern with X API"); it is rescued by the strong preceding paragraph (API inserts quant/dequant, gradients flow, records config). Borderline ✅, but the listing gets **no post-listing walk-through** of the specific API calls (`prepare_qat` → `convert`), so a reader is left to map prose→code unaided.
- **Fix (suggested, not applied):** add one sentence after the listing naming what to notice, e.g. *"The `prepare_qat` call inserts the fake-quant observers; `convert` then freezes them into the integer graph — the rest of the training loop is unchanged."* Or fold the specifics into the ref sentence.

### `lst-pytorch_pruning` — line 7401
- **Caption:** PyTorch Pruning APIs: unstructured + structured pruning via `torch.nn.utils.prune`.
- **Ref (L7399):** "The same framework-level pattern applies to pruning. The API owns the weight tensor, applies a mask or structured removal rule… @Lst-pytorch_pruning illustrates both unstructured and structured pruning."
- **Verdict:** ✅ Explained — preceding sentence gives the mechanism (mask vs structured removal); post-listing prose (L7418) delivers the "repeatability" payoff.
- **Fix:** none.

---

## Pilot summary

| Verdict | Count |
|---|---|
| ✅ Explained | 5 |
| ⚠️ Announcer (mild) | 1 |
| 🛑 Bare-pointer / orphan | 0 |

**Hit rate: 1/6, and that one is borderline.** This chapter's listings are in good shape. The audit's value here is mostly confirmation, which is the right behavior — it does not manufacture findings on healthy prose. The payoff will concentrate in chapters where the ref sentence is thin AND no surrounding prose explains what to notice.
