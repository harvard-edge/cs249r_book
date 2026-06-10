# Float exposition eval — nn_computation.qmd (vol1)
Standard: FLOAT_EXPOSITION_STANDARD.md (caption excluded from prose budget)

## Summary
| type | level | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|---|
| equation | 🔴 | 27 | 17 | 10 | 0 |
| algorithm | 🔴 | 2 | 2 | 0 | 0 |
| table | 🟠 | 15 | 8 | 7 | 0 |
| figure | 🟠 | 20 | 14 | 6 | 0 |
| listing | 🟡 | 0 | — | — | — |
| **total** | | **64** | **41** | **23** | **0** |

---

## Findings (⚠️ only)

### EQUATIONS

---

### ⚠️ `eq-sigmoid` (equation 🔴) — def L2430
- **Ref (body prose):** "The sigmoid function maps any input value to a bounded range between 0 and 1, as defined in @eq-sigmoid:"
- **Missing move:** lead-out; the consequence of the bounded range (saturation, vanishing gradient) lives only in footnote `[^fn-sigmoid-etymology]`, not in body prose. After the equation the next body text is the footnote marker.
- **Suggested rewrite (no em-dash/hyphen, ≤1 colon/para, content leads):**
  ```diff
  - The sigmoid function maps any input value to a bounded range between 0 and 1, as defined in @eq-sigmoid:
  + The sigmoid function maps any input value to a bounded range between 0 and 1, as defined in @eq-sigmoid. The
  + bounded output means the function saturates near 0 and 1, where its gradient approaches zero; in deep networks,
  + multiplying many near-zero gradients across layers shrinks the update signal exponentially, a phenomenon known
  + as the vanishing gradient problem.
  ```

---

### ⚠️ `eq-relu` (equation 🔴) — def L2458
- **Ref (body prose):** "The ReLU function is defined in @eq-relu:"
- **Missing move:** lead-out; the consequence of `max(0, x)` — why it avoids vanishing gradients and its hardware cost advantage — lives only in footnote `[^fn-relu-hardware-efficiency]`, not in body prose. The body after the equation goes directly to the footnote marker.
- **Suggested rewrite:**
  ```diff
  - The ReLU function is defined in @eq-relu:
  + The ReLU function is defined in @eq-relu. Because the derivative of max(0, x) is 1 for positive inputs and 0
  + otherwise, the gradient never shrinks through a ReLU unit when the input is positive, eliminating the
  + saturation that stalls sigmoid-based training. This also makes ReLU trivially cheap to evaluate: a single
  + comparison instruction, versus the floating-point exponential sigmoid requires.
  ```

---

### ⚠️ `eq-softmax` (equation 🔴) — def L2473
- **Ref (body prose, first citation):** "The softmax function is defined in @eq-softmax:"
- **Missing move:** lead-out for the first citation; the implication (numerical stability hazard, overflow for inputs > ~88) lives only in footnote `[^fn-softmax-etymology]`. The body prose following the equation goes to the footnote marker, then a subsequent paragraph narrates the vector-wise nature and logit concept — but never states the consequence that motivates the stable-softmax variant introduced later.
- **Suggested rewrite:**
  ```diff
  - The softmax function is defined in @eq-softmax:
  + The softmax function is defined in @eq-softmax. Because every output depends on all inputs through the
  + denominator, softmax is not an element-wise operation: the network must evaluate the entire logit vector
  + before any probability is known. This vector-level dependency also introduces a numerical hazard: the
  + exponential in the numerator overflows standard 32-bit floats for inputs above roughly 88, a common
  + source of silent NaN failures that motivates the numerically stable variant introduced later.
  ```

---

### ⚠️ `eq-layer-activation` (equation 🔴) — def L2856
- **Ref (body prose):** "This gives us the complete layer computation in @eq-layer-activation:"
- **Missing move:** lead-out; the payoff paragraph (L2858) only names the symbol conventions and says "core architecture concepts are complete enough to proceed." It does not state what the equation implies: that every layer's output is the result of composing a linear map with a nonlinear gate, and that this composition is what gives the network depth-wise expressiveness distinct from a single wide layer.
- **Suggested rewrite:**
  ```diff
  - Where $\mathbf{a}^{(\ell)}$ (written as $\mathbf{A}^{(\ell)}$ for batches) represents the layer's activation
  - output. We adopt the row-vector convention throughout: each sample is a row, and the weight matrix
  - $\mathbf{W}^{(\ell)} \in \mathbb{R}^{n_{\ell-1} \times n_\ell}$ maps from the previous layer's width to the
  - current layer's width. With this equation in place, the core architecture concepts are complete enough to
  - proceed.
  + Where $\mathbf{a}^{(\ell)}$ represents the layer's activation output (written $\mathbf{A}^{(\ell)}$ for
  + batches), and the weight matrix $\mathbf{W}^{(\ell)} \in \mathbb{R}^{n_{\ell-1} \times n_\ell}$ maps from
  + the previous layer's width to the current layer's width using the row-vector convention. The key consequence
  + is that each layer applies a different linear transformation followed by a different nonlinearity: stacking
  + these operations lets the network build increasingly abstract representations in a way that any single
  + linear map cannot, regardless of width.
  ```

---

### ⚠️ `eq-loss-general` (equation 🔴) — def L3676
- **Ref (body prose):** "The network's error is measured by a loss function $\mathcal{L}$, as shown in @eq-loss-general:"
- **Missing move:** lead-out; the implication of $\mathcal{L}(\hat{y}, y)$ as a gradient landscape (what shapes learning, flat regions, steep regions) lives only in footnote `[^fn-loss-function]`. The body after the equation continues narrating training vs. inference semantics of $\theta$, not what the loss function does to optimization.
- **Suggested rewrite:**
  ```diff
  - The network's error is measured by a loss function $\mathcal{L}$, as shown in @eq-loss-general:
  + The network's error is measured by a loss function $\mathcal{L}$, as shown in @eq-loss-general. The choice
  + of $\mathcal{L}$ determines the gradient landscape that training must navigate: a loss with flat regions near
  + incorrect predictions produces weak gradients that stall learning, while a loss with steep gradients near
  + the decision boundary guides the optimizer efficiently. The specific form of $\mathcal{L}$ for classification
  + is developed below; the critical property is that it must be differentiable with respect to $\hat{y}$ so
  + backpropagation can compute update signals.
  ```

---

### ⚠️ `eq-layer-activation-transform` (equation 🔴) — def L3855
- **Ref (body prose):** "Following this linear transformation, each layer applies a nonlinear activation function $f$... as expressed in @eq-layer-activation-transform:"
- **Missing move:** lead-out; the payoff (L3857) says only "this process repeats at each layer, creating a chain of transformations," then shows a chain-diagram equation. It does not state the consequence: that applying $f$ after each linear step is the operation that prevents the network from collapsing to a single linear map and is what makes depth computationally meaningful.
- **Suggested rewrite:**
  ```diff
  - This process repeats at each layer, creating a chain of transformations:
  + Applying $f$ at each layer is the operation that prevents depth from being redundant: without it, composing
  + any number of linear transforms produces exactly one linear transform, making additional layers useless.
  + With it, each layer can carve new nonlinear boundaries in representation space. The resulting chain of
  + alternating linear and nonlinear steps is:
  ```

---

### ⚠️ `eq-batch-cross-entropy` (equation 🔴) — def L4168
- **Ref (body prose):** "For a batch of $B$ examples, the cross-entropy loss becomes @eq-batch-cross-entropy:"
- **Missing move:** lead-out; after the equation the prose pivots immediately to numerical stability (the NaN risk from log(0)), which concerns the implementation, not the meaning of the double-sum formula. The implication of the double sum — that it averages over examples and sums over classes, and why averaging makes gradient scale independent of batch size — is not stated.
- **Suggested rewrite:**
  ```diff
  - For a batch of $B$ examples, the cross-entropy loss becomes @eq-batch-cross-entropy:
  + For a batch of $B$ examples, the cross-entropy loss becomes @eq-batch-cross-entropy. The outer sum averages
  + over examples and the inner sum selects the correct-class contribution via one-hot $y_{ij}$: because
  + division by $B$ normalizes the result, the same learning rate works regardless of batch size, and the
  + per-example terms are computed independently before a single reduction, mapping directly to parallel
  + hardware.
  ```

---

### ⚠️ `eq-epsilon-stability` (equation 🔴) — def L4175
- **Ref (body prose):** "1. Add a small epsilon to prevent taking log of zero, as in @eq-epsilon-stability:"
- **Missing move:** lead-out; the equation is dropped as a numbered list item with no stated consequence beyond "prevent taking log of zero." The tradeoff (epsilon introduces a small bias into the gradient — a production engineering decision) is not named.
- **Suggested rewrite:**
  ```diff
  - 1. Add a small epsilon to prevent taking log of zero, as in @eq-epsilon-stability:
  + 1. Clip the probability away from zero by adding a small constant $\epsilon$ before taking the log, as in
  +    @eq-epsilon-stability. This prevents $-\infty$ loss from a zero probability, at the cost of a small
  +    constant bias in the gradient — an acceptable engineering tradeoff in production training pipelines.
  ```

---

### ⚠️ `eq-softmax-stable` (equation 🔴) — def L4178
- **Ref (body prose):** "2. Apply the log-sum-exp trick for numerical stability... as shown in @eq-softmax-stable:"
- **Missing move:** lead-out; the equation is dropped as a numbered list item with an appendix reference carrying the explanation. The body prose does not state WHY subtracting max($z$) stabilizes computation (it shifts all inputs to be non-positive, capping the exponential below 1 so no overflow occurs while leaving the output probabilities unchanged because the subtraction cancels in numerator and denominator).
- **Suggested rewrite:**
  ```diff
  - 2. Apply the log-sum-exp trick for numerical stability (see @sec-appdx-data-foundations-logits-numerical-stability-13e2
  -    for why this is necessary and how it works), as shown in @eq-softmax-stable:
  + 2. Shift all logits by the maximum value before exponentiating, as shown in @eq-softmax-stable. Subtracting
  +    max($z$) makes every exponentiated value at most 1, eliminating overflow, while leaving the probabilities
  +    unchanged because the shift cancels in numerator and denominator. This is the standard numerically stable
  +    softmax implementation used in all major frameworks. See @sec-appdx-data-foundations-logits-numerical-stability-13e2
  +    for the full derivation.
  ```

---

### ⚠️ `eq-weight-gradient` (equation 🔴) — def L4301
- **Ref (body prose):** "These gradients tell us precisely how to adjust the connection strengths between neurons to reduce prediction errors, as shown in @eq-weight-gradient:"
- **Missing move:** lead-out; the citation sentence names the purpose but does not interpret the specific mathematical form. Why $\mathbf{A}^{(\ell-1)T}$ appears on the left — that the outer product between incoming activations and the upstream gradient is what accumulates the update direction — is not stated. The payoff paragraph introduces the bias gradient equation, not the implication of the weight gradient form.
- **Suggested rewrite:**
  ```diff
  - Weight gradients measure how changing each weight affects the final loss. These gradients tell us precisely
  - how to adjust the connection strengths between neurons to reduce prediction errors, as shown in
  - @eq-weight-gradient:
  + Weight gradients measure how changing each weight affects the final loss, as shown in @eq-weight-gradient.
  + The transpose ${\mathbf{A}^{(\ell-1)}}^T$ appears because each weight $W_{ij}$ connects input activation
  + $A_i$ to output preactivation $Z_j$: the gradient is the outer product of the incoming activations with the
  + upstream error signal, so heavily activated inputs receive proportionally larger gradient updates.
  ```

---

### ⚠️ `eq-bias-gradient` (equation 🔴) — def L4304
- **Ref (body prose):** "Since biases shift the activation threshold of neurons, these gradients indicate whether neurons should become more or less easily activated, as expressed in @eq-bias-gradient:"
- **Missing move:** lead-out; the payoff paragraph moves directly to the activation gradient equation. The specific form ($\mathbf{1}^T$ multiplied by the upstream gradient) is not explained: it sums the upstream gradient over the batch dimension because biases are shared across examples, so their update is the mean over the batch.
- **Suggested rewrite:**
  ```diff
  - Since biases shift the activation threshold of neurons, these gradients indicate whether neurons should
  - become more or less easily activated, as expressed in @eq-bias-gradient:
  + Since biases are shared across all examples in the batch, the bias gradient sums the upstream error
  + $\partial\mathcal{L}/\partial\mathbf{Z}^{(\ell)}$ over every example before the optimizer step, which is
  + what the $\mathbf{1}^T$ multiplication achieves in @eq-bias-gradient. A large accumulated bias gradient
  + indicates that the neuron's threshold is consistently wrong across the batch, not just for a single example.
  ```

---

### FIGURES

---

### ⚠️ `fig-ai-ml-dl` (figure 🟠) — def L159
- **Ref (body prose):** "examine the concentric layers in @fig-ai-ml-dl: neural networks sit at the core of deep learning, which is itself a subset of machine learning, which falls under the umbrella of artificial intelligence."
- **Missing move:** lead-out; the payoff paragraph (L382) is far downstream and pivots to gradient instabilities. The immediate prose after the figure and citation never states the takeaway: that this containment hierarchy implies a containment of scope (every neural network is a machine learning model but not vice versa) or the systems consequence (deep learning inherits all of ML's data and optimization requirements plus its own hardware demands).
- **Suggested rewrite:**
  ```diff
  - Classical machine learning required human experts to design feature extractors for each new problem...
  - examine the concentric layers in @fig-ai-ml-dl: neural networks sit at the core of deep learning, which is
  - itself a subset of machine learning, which falls under the umbrella of artificial intelligence.
  + Classical machine learning required human experts to design feature extractors for each new problem...
  + examine the concentric layers in @fig-ai-ml-dl: neural networks sit at the core of deep learning, which is
  + itself a subset of machine learning, which falls under the umbrella of artificial intelligence. This
  + containment has a practical implication: deep learning inherits all of machine learning's requirements for
  + labeled data, optimization, and evaluation, and then adds its own demands for large-scale parallel compute
  + and automatic differentiation. Solving a deep learning problem therefore requires solving the machine
  + learning problem underneath it.
  ```

---

### ⚠️ `fig-breakout` (figure 🟠) — def L475
- **Ref (body prose):** "The program needs explicit rules for every interaction... the brick should be removed and the ball's direction should be reversed (@fig-breakout). While this approach works effectively for games with clear physics and limited states, it hits a wall when dealing with the messy, unstructured data of the real world."
- **Missing move:** lead-out; the citation sentence names the limitation in passing ("hits a wall") but the payoff (L538) pivots away to the broader traditional-programming paradigm diagram without closing the Breakout case. The specific takeaway from the figure — that even this simple game requires branching code for every collision case, making rule sets grow combinatorially — is not stated in prose.
- **Suggested rewrite:**
  ```diff
  - it hits a wall when dealing with the messy, unstructured data of the real world.
  + it hits a wall when dealing with the messy, unstructured data of the real world. The Breakout figure
  + reveals the structural problem: even a single brick collision demands a separate conditional for detection,
  + removal, and direction reversal. Real-world perception tasks have millions of such cases, and no finite
  + set of rules can enumerate them.
  ```

---

### ⚠️ `fig-activity-rules` (figure 🟠) — def L574
- **Ref (body prose):** "Speed variations, transitions between activities, and boundary cases each demand additional rules, creating unwieldy decision trees (@fig-activity-rules)."
- **Missing move:** lead-out; the parenthetical citation is followed immediately by a sentence about cat detection (pivoting away). The takeaway from the figure — that the decision tree grows in depth and branch count for each edge case, making maintenance cost superlinear — is not stated.
- **Suggested rewrite:**
  ```diff
  - Speed variations, transitions between activities, and boundary cases each demand additional rules, creating
  - unwieldy decision trees (@fig-activity-rules). Computer vision tasks compound these difficulties.
  + Speed variations, transitions between activities, and boundary cases each demand additional rules, creating
  + unwieldy decision trees as shown in @fig-activity-rules: each new edge case adds a branch, and branches
  + multiply exponentially with the number of dimensions under consideration. Computer vision tasks compound
  + these difficulties.
  ```

---

### ⚠️ `fig-hog` (figure 🟠) — def L660
- **Ref (body prose):** "The Histogram of Oriented Gradients (HOG) method exemplifies this approach... This transforms raw pixels into shape descriptors robust to lighting variations and small positional changes."
- **Missing move:** lead-out; the payoff lives only in footnote `[^fn-hog-method]`. Body prose after the figure goes to the footnote marker. The takeaway — that HOG's fixed descriptor width still requires tuning per domain, limiting generality — is not in running body prose.
- **Suggested rewrite:**
  ```diff
  - This transforms raw pixels into shape descriptors robust to lighting variations and small positional
  - changes.
  + This transforms raw pixels into shape descriptors robust to lighting variations and small positional
  + changes. The figure shows the three-stage pipeline: detect edges, divide the image into fixed cells, and
  + histogram the edge orientations per cell. The limitation is that the cell size and orientation bins are
  + fixed hyperparameters chosen by a human engineer, so the descriptor that works for pedestrian detection
  + requires redesign for face recognition or vehicle identification — a fundamental constraint that learned
  + representations eliminate.
  ```

---

### ⚠️ `fig-connections` (figure 🟠) — def L2752
- **Ref (body prose):** "@Fig-connections makes the dense pattern explicit by laying out a small three-layer network with every connection weight labeled. Every input connects to every hidden neuron... Each labeled edge represents one learnable weight, making visible the total parameter count and, consequently, why matrix multiplication dominates neural network computation: the weight matrix dimensions directly determine both the layer's storage requirements and its arithmetic cost."
- **Missing move:** lead-out; the citation sentence is reasonably strong, but the payoff paragraph (L2838) moves to bias terms without closing the key observation the figure demonstrates: that parameter count scales as the product of adjacent layer widths, making wide layers expensive in a way the labeled edges make concrete.
- **Suggested rewrite:**
  ```diff
  - Each neuron in a layer also has an associated bias term. While weights determine the relative importance of
  - inputs, biases allow neurons to shift their activation functions.
  + The labeled-weight layout in @fig-connections makes the quadratic scaling visible: a layer with $n_1$ inputs
  + and $n_2$ neurons requires $n_1 \times n_2$ weight entries. Each neuron also has an associated bias term
  + that shifts its activation threshold.
  ```

---

### ⚠️ `fig-usps-digit-examples` (figure 🟠) — def L5105
- **Ref (body prose):** "The samples in @fig-usps-digit-examples show the wide variation in writing styles, pen types, stroke thickness, and character formation that the system must handle."
- **Missing move:** lead-out; the payoff (L5111) says only "the challenging environment imposed requirements spanning every aspect of neural network implementation discussed in this chapter." This is a summary pivot, not an interpretation of the figure. The figure's takeaway — that these variations represent the gap between the clean MNIST training distribution and production mail images, making generalization the binding challenge — is not stated.
- **Suggested rewrite:**
  ```diff
  - The system must make accurate predictions quickly enough to maintain mail processing speeds, yet errors in
  - recognition can lead to significant delays and costs from misrouted mail.
  + The system must make accurate predictions quickly enough to maintain mail processing speeds, yet errors
  + cause delays from misrouted mail. The figure shows why the gap between MNIST's clean training digits and
  + real USPS samples is the central challenge: stroke widths span an order of magnitude, slant varies
  + continuously, and some characters are indistinguishable from adjacent classes without context. A model
  + that achieves high accuracy on MNIST training data must generalize across this variation to be deployable.
  ```

---

### ⚠️ `fig-usps-inference-pipeline` (figure 🟠) — def L5129
- **Ref (body prose):** "Trace the data flow in @fig-usps-inference-pipeline to see this hybrid architecture in action, with the neural network operating as one component within a broader pipeline of conventional preprocessing and postprocessing stages."
- **Missing move:** lead-out; after the figure the prose narrates the imaging station (preprocessing) in detail but never closes on what the three-color pipeline architecture demonstrates: that the neural network handles only the pattern-recognition step while classical components handle the structured operations before and after it, and that this division of labor is a general architectural principle for production ML systems.
- **Suggested rewrite:**
  ```diff
  - The process begins when an envelope reaches the imaging station.
  + The architecture in @fig-usps-inference-pipeline captures a principle that recurs across production ML
  + deployments: the neural network handles the pattern-recognition step that defies rule-based programming,
  + while deterministic classical components handle structured operations on both sides. Preprocessing converts
  + raw sensor data into a clean, normalized form the network can consume; postprocessing converts raw network
  + outputs into actionable decisions the physical system can execute. The process begins when an envelope
  + reaches the imaging station.
  ```

---

### TABLES

---

### ⚠️ `tbl-nn-computation-mnist-params` (table 🟠) — def L3387
- **Ref (body prose):** "**Step 1**: Model parameters. @Tbl-nn-computation-mnist-params tallies the weights and biases layer by layer:"
- **Missing move:** lead-out; the citation is a bare "Step 1" pointer with no extracted insight. The payoff (L3408) immediately cites the next table without stating what the params table reveals: that the first layer dominates the parameter budget because it must connect the full 784-dimensional input, and that this first-layer dominance is a general property of fully connected networks applied to high-dimensional inputs.
- **Suggested rewrite:**
  ```diff
  - **Step 1**: Model parameters. @Tbl-nn-computation-mnist-params tallies the weights and biases layer by layer:
  + **Step 1**: Model parameters. @Tbl-nn-computation-mnist-params tallies the weights and biases layer by
  + layer. The first layer dominates: connecting 784 inputs to 128 neurons requires roughly 10$\times$ more
  + parameters than either subsequent layer, because parameter count scales with the product of adjacent layer
  + widths. This first-layer dominance is a general property of fully connected networks applied to
  + high-dimensional inputs.
  ```

---

### ⚠️ `tbl-nn-computation-mnist-activations` (table 🟠) — def L3401
- **Ref (body prose):** "**Step 2**: Activations. @Tbl-nn-computation-mnist-activations records each layer's activation tensor and its memory cost at batch size 32:"
- **Missing move:** lead-out; the citation is a bare "Step 2" pointer. The payoff (L3408) immediately cites the memory-budget table. The table's insight — that activation memory scales with batch size (each of the 32 examples needs its own copy of every intermediate representation) — is not stated.
- **Suggested rewrite:**
  ```diff
  - **Step 2**: Activations. @Tbl-nn-computation-mnist-activations records each layer's activation tensor and
  - its memory cost at batch size 32:
  + **Step 2**: Activations. @Tbl-nn-computation-mnist-activations records each layer's activation tensor and
  + its memory cost at batch size 32. Activation memory scales linearly with batch size: processing 64 images
  + instead of 32 doubles this footprint while the parameter footprint stays constant. For this small MNIST
  + network the activations remain below the parameter budget, but the table makes clear why at large batch
  + sizes or in deeper networks activations become the dominant memory consumer.
  ```

---

### ⚠️ `tbl-nn-computation-mnist-memory-budget` (table 🟠) — def L3418
- **Ref (body prose):** "@Tbl-nn-computation-mnist-memory-budget summarizes the per-component memory footprint for training vs. inference."
- **Missing move:** lead-out; bare pointer with no extracted conclusion. The payoff (L3450) moves to parameter count growth in a different context. The table's key result — that training memory is dominated by the gradient and optimizer-state components that are absent from inference, explaining the several-times multiplier — is not stated in prose.
- **Suggested rewrite:**
  ```diff
  - @Tbl-nn-computation-mnist-memory-budget summarizes the per-component memory footprint for training vs.
  - inference.
  + @Tbl-nn-computation-mnist-memory-budget summarizes the per-component memory footprint for training vs.
  + inference. The dominant gap is the gradient and optimizer-state rows, which are present only during
  + training: Adam stores momentum and velocity vectors that together equal twice the parameter count, so even
  + a modest model accumulates several times more training memory than inference memory for the same
  + architecture and batch size.
  ```

---

### ⚠️ `tbl-nn-computation-napkin-math-checks` (table 🟠) — def L3625
- **Ref (body prose):** "@Tbl-nn-computation-napkin-math-checks distills three of the most common feasibility questions into one-line formulas an engineer can apply before reaching for a profiler."
- **Missing move:** lead-out; the citation names the table's purpose but does not state the load-bearing implication: which of the three checks tends to be the binding constraint first and why (GPU-fit check fails before throughput check for large models because memory capacity is harder to increase than compute throughput). The payoff (L3627) pivots to parameter distribution instead.
- **Suggested rewrite:**
  ```diff
  - @Tbl-nn-computation-napkin-math-checks distills three of the most common feasibility questions into
  - one-line formulas an engineer can apply before reaching for a profiler.
  + @Tbl-nn-computation-napkin-math-checks distills three feasibility questions into formulas an engineer can
  + apply before profiling. The GPU-fit check typically fails first: accelerator memory capacity is a hard
  + ceiling that cannot be traded against compute, while epoch time and bound regime can often be improved by
  + changing batch size, precision, or pipeline depth. Checking fit before estimating throughput saves
  + expensive profiling runs on architectures that cannot be loaded at all.
  ```

---

### ⚠️ `tbl-nn-computation-mnist-flops` (table 🟠) — def L4097
- **Ref (body prose):** "**Solution**: @Tbl-nn-computation-mnist-flops breaks down the operation count layer by layer. The total comes to ~`{python} MnistFlopsCalc.total_mops_str` MOp, or ... ~`{python} MnistFlopsCalc.per_image_kops_str` KOp per image."
- **Missing move:** lead-out; the citation states the total but not the implication. The payoff (L4101) says only "Forward propagation is easy to state mathematically, but its implementation is constrained by activation storage, batch size, memory layout, and hardware fit" — a generic framing that does not name what the table demonstrates. The key result — that the first-layer matrix multiply accounts for the overwhelming majority of the operation count and that bias/activation rows are negligible by comparison — is not stated.
- **Suggested rewrite:**
  ```diff
  - **Solution**: @Tbl-nn-computation-mnist-flops breaks down the operation count layer by layer.
  + **Solution**: @Tbl-nn-computation-mnist-flops breaks down the operation count layer by layer. The first
  + layer's matrix multiply dominates: connecting 784 inputs to 128 hidden units accounts for most of the
  + total operation count, while bias additions and ReLU comparisons together contribute under 1 percent.
  + This concentration of arithmetic in one operation type — matrix multiplication — is why hardware
  + accelerators optimized for dense matrix math dominate neural network inference.
  ```

---

### ⚠️ `tbl-usps-numbers` (table 🟠) — def L5562
- **Ref (body prose):** "@Tbl-usps-numbers summarizes the key performance metrics."
- **Missing move:** bare pointer with no extracted conclusion. The payoff (L5570) narrates accuracy variation with real-world factors but never names the specific numbers from the table that constitute the argument (1 percent neural network error vs. 2.5 percent human error; 10 to 30 times faster throughput; 9 percent rejection rate as the economic optimum). The table's takeaway — that the neural network not only matched but surpassed human performance while running faster, validating the deployment decision — should be in prose.
- **Suggested rewrite:**
  ```diff
  - Neural network-based ZIP code recognition transformed USPS mail processing operations. By 2000, several
  - facilities across the country used this technology, processing millions of mail pieces daily. This
  - real-world deployment demonstrated both the potential and the limitations of neural networks in
  - mission-critical applications. @Tbl-usps-numbers summarizes the key performance metrics.
  + Neural network-based ZIP code recognition transformed USPS mail processing operations. By 2000, several
  + facilities across the country used this technology, processing millions of mail pieces daily.
  + @Tbl-usps-numbers summarizes the key results: the network achieved roughly 1 percent digit error versus
  + 2.5 percent for human operators, while processing digits 10 to 30 times faster. The 9 percent rejection
  + rate represents the economically optimal threshold where the cost of a misrouted letter exceeds the cost
  + of routing that letter to human review. Together these numbers confirmed that neural networks could
  + surpass human performance on constrained pattern-recognition tasks even with 1989-era hardware.
  ```
