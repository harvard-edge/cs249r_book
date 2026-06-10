# Float-explanation worklist — nn_architectures.qmd (vol1)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 14 | 11 | 3 | 0 |
| table | 15 | 15 | 0 | 0 |
| listing | 8 | 8 | 0 | 0 |
| algorithm | 0 | 0 | 0 | 0 |
| equation | 12 | 12 | 0 | 0 |
| **total** | **49** | **46** | **3** | **0** |

---

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

### ⚠️ `fig-mlp` — def L847  (Thin)
- **Caption:** Rich — describes three-layer architecture, $\mathcal{O}(N \times M)$ connectivity, MNIST GEMM cost.
- **Ref(s):** L845 `@Fig-mlp`: "Dense connectivity translates directly into fully connected layers and matrix multiplication operations, the mathematical basis introduced in @sec-neural-computation-matrix-multiplication-formulation-417c that makes MLPs computationally tractable. @Fig-mlp shows how each layer transforms its input through this core operation."
- **Context checked:** ref ⚠️ (float-announcer) · prev ¶ ✓ (explains dense connectivity) · next ¶ is the float def · caption ✓ (carries the content) · payoff ✓ (L1001 unpacks the equation)
- **Issue:** The reference sentence is a bare "shows how" pointer. The caption does the explanatory work. The prose reference should carry at least one concrete takeaway so readers who scan the text gain the insight without stopping at the figure.
- **Suggested rewrite (flag-only):**
  ```diff
  - Dense connectivity translates directly into fully connected layers\index{Fully-Connected Layer} and matrix multiplication operations, the mathematical basis introduced in @sec-neural-computation-matrix-multiplication-formulation-417c that makes MLPs computationally tractable. @Fig-mlp shows how each layer transforms its input through this core operation.
  + Dense connectivity translates directly into fully connected layers\index{Fully-Connected Layer} and matrix multiplication operations, the mathematical basis introduced in @sec-neural-computation-matrix-multiplication-formulation-417c that makes MLPs computationally tractable. In @fig-mlp every neuron in each layer connects to every neuron in the next, so a single 784-to-100 hidden layer requires a $784{\times}100$ weight matrix and 78,400 multiply-accumulate operations per sample — the dense all-to-all pattern that makes MLPs bandwidth-bound at inference.
  ```

---

### ⚠️ `fig-cnn-spatial-processing` — def L1237  (Thin — placement mismatch)
- **Caption:** Explains hierarchical feature extraction and translation equivariance through learnable filters.
- **Ref(s):** L1455 `@fig-cnn-spatial-processing`: "As @fig-cnn-spatial-processing illustrates, convolutional neural networks meet both requirements through hierarchical feature extraction, where simple patterns compose into increasingly complex representations at successive layers."
- **Context checked:** ref ✓ (L1455 explains what the figure shows) · prev ¶ ✓ · caption ✓ · payoff ✓
- **Issue:** The figure is *defined* at L1237 (in the "Pattern processing needs" section opener, immediately after one setup sentence) with no preceding prose reference. The first `@fig-cnn-spatial-processing` pointer appears 218 lines later at L1455, in a different subsection. A reader encountering the figure at its placement position has no pointer telling them what to look for. The explanation itself, when it arrives at L1455, is adequate — the problem is purely positional.
- **Suggested fix (flag-only):** Add a `@fig-cnn-spatial-processing` forward pointer to the paragraph at L1235 (the section opener), so the figure is introduced before it appears:
  ```diff
  - Spatial pattern processing addresses scenarios where the relationship between data points depends on their relative positions or proximity. Consider processing a natural image: a pixel's relationship with its neighbors is important for detecting edges, textures, and shapes. These local patterns then combine hierarchically to form more complex features: edges form shapes, shapes form objects, and objects form scenes.
  + Spatial pattern processing addresses scenarios where the relationship between data points depends on their relative positions or proximity. Consider processing a natural image: a pixel's relationship with its neighbors is important for detecting edges, textures, and shapes. These local patterns then combine hierarchically to form more complex features, as @fig-cnn-spatial-processing illustrates: edges form shapes, shapes form objects, and objects form scenes, with each CNN layer extracting progressively more abstract representations of the input.
  ```

---

### ⚠️ `fig-transformer-attention-visualized` — def L2348  (Thin — placement mismatch)
- **Caption:** Describes an attention head resolving the pronoun "they," with line thickness encoding attention weight magnitude.
- **Ref(s):** L2398 `@fig-transformer-attention-visualized`: "To see attention in action, consider @fig-transformer-attention-visualized. When processing the pronoun 'they' in the sentence, the attention mechanism must determine what 'they' refers to. The attention weights (indicated by line thickness) emphasize 'student' and 'finish'..."
- **Context checked:** ref ✓ (L2398 is detailed and explanatory) · prev ¶ L2346 ✗ (sets up the concept but does not point to the figure) · caption ✓ · payoff: no post-float prose before next subsection
- **Issue:** The figure is defined at L2348, in the "Pattern processing needs" section opener immediately after a motivating paragraph (L2346) that does not reference it. The explanatory reference at L2398 is 50 lines later, appearing after an extended multi-domain discussion (GCNs, protein folding, document analysis). A reader encountering the figure at its printed position has no local prose pointer. The explanation is genuinely good once it arrives; the problem is that it arrives late.
- **Suggested fix (flag-only):** Add a forward pointer to the motivating paragraph at L2346, so the figure is introduced before it appears:
  ```diff
  - Dynamic pattern processing addresses scenarios where relationships between elements are not fixed by architecture but instead emerge from content. Language translation exemplifies this challenge: when translating "the bank by the river," understanding "bank" requires attending to "river," but in "the bank approved the loan," the important relationship is with "approved" and "loan." Unlike RNNs that process information sequentially or CNNs that use fixed spatial patterns, an architecture is required that can dynamically determine which relationships matter.
  + Dynamic pattern processing addresses scenarios where relationships between elements are not fixed by architecture but instead emerge from content. Language translation exemplifies this challenge: when translating "the bank by the river," understanding "bank" requires attending to "river," but in "the bank approved the loan," the important relationship is with "approved" and "loan." As @fig-transformer-attention-visualized shows for a concrete sentence, the correct attendee for each token depends entirely on input content, not position. Unlike RNNs that process information sequentially or CNNs that use fixed spatial patterns, an architecture is required that can dynamically determine which relationships matter.
  ```
