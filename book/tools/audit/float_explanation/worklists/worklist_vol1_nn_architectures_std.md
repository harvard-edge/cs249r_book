# Float Exposition Worklist — `nn_architectures.qmd` (vol1)

Audited against: `FLOAT_EXPOSITION_STANDARD.md`
Caption/alt-text/code comments do NOT count toward the prose's job.
Only running body prose (citation sentence + narrative before/after) is graded.

---

## Summary Table

| Type      | Level    | Total | ✅ Meets | ⚠️ Partial | 🛑 Fails |
|:----------|:---------|------:|--------:|----------:|--------:|
| Equation  | Strict   |    12 |       8 |         4 |       0 |
| Figure    | High     |    14 |       8 |         6 |       0 |
| Listing   | Medium   |     8 |       7 |         1 |       0 |
| Table     | High     |    15 |      10 |         5 |       0 |
| **Total** |          |    49 |      33 |        16 |       0 |

**16 findings, all ⚠️ Partial. Dominant type: Figure (6) and Table (5), with Equation (4) and Listing (1) also contributing.**

---

## ✅ Passing Floats (no action needed)

| Label | Type | Verdict |
|:------|:-----|:--------|
| `eq-dense-layer` | Equation | ✅ Symbols fully glossed; consequence stated in payoff |
| `eq-dense-intensity` | Equation | ✅ Derivation inline; memory-bound consequence explicit |
| `eq-equivariance` | Equation | ✅ Defined with concrete five-pixel example immediately after |
| `eq-invariance` | Equation | ✅ Contrasted with equivariance; consequence (loss of relational info) stated |
| `eq-batchnorm-normalize` | Equation | ✅ Paired with `eq-batchnorm-transform`; both cited with functional split; payoff gives identity-transform preservation insight |
| `eq-batchnorm-transform` | Equation | ✅ (same cite paragraph as above) |
| `eq-layernorm` | Equation | ✅ Payoff delivers per-sample independence and why transformers adopt it |
| `eq-residual` | Equation | ✅ Inline "where" clause; gradient-path split ("two paths") stated in prose two lines later |
| `fig-efficiency-frontier` | Figure | ✅ Second cite (L321) delivers three-era Pareto reading |
| `fig-transformer-attention-visualized` | Figure | ✅ Cite narrates what the model attends to and why |
| `fig-transformer` | Figure | ✅ Cite traces encoder→decoder data flow end-to-end |
| `fig-context-explosion` | Figure | ✅ Cite delivers quantitative reading (flat 2018–2022, 3× orders of magnitude growth) and KV-cache trade-off |
| `fig-im2col-diagram` | Figure | ✅ Both cites carry the transformation story and memory-for-GEMM trade-off |
| `fig-collective-comm` | Figure | ✅ Cite names all four patterns with system examples; scatter/gather/reduce unpacked in following paragraphs |
| `fig-dnn-fm-framework` | Figure | ✅ Cite interprets the flowchart's top-to-bottom logic and DLRM special case |
| `lst-mlp_layer_matrix` | Listing | ✅ Payoff paragraph (L1100) delivers "single-line matmul abstracts O(N×M) complexity" |
| `lst-mlp_layer_compute` | Listing | ✅ Cite names nested loops and three-tier decomposition; payoff delivers BLAS utilization note |
| `lst-conv_layer_spatial` | Listing | ✅ Cite frames framework-abstraction vs. hardware reality; payoff points to im2col explanation |
| `lst-conv_layer_compute` | Listing | ✅ Cite immediately interprets seven nested loops and why im2col supersedes them |
| `lst-rnn_layer_compute` | Listing | ✅ Cite frames "computational reality beneath mathematical abstraction"; payoff walks the loop groups |
| `lst-attention_layer_compute` | Listing | ✅ Cite names pairwise attention scoring; payoff (L2820) delivers full O(S²) signature |
| `lst-self_attention_layer` | Listing | ✅ Payoff (L3227) states parallel processing during training and pivot to inference |
| `tbl-workload-signatures` | Table | ✅ Second cite (L588) delivers full intensity-spectrum explanation; third cite applies it |
| `tbl-lighthouse-comparison` | Table | ✅ First cite names the "Bottleneck" column key; second cite connects to energy profiles |
| `tbl-nn-architectures-resnet50-profile` | Table | ✅ Payoff (L1971) states compute-bound ceiling and hardware-fit consequence |
| `tbl-nn-architectures-mobilenet-profile` | Table | ✅ Payoff (L1998) delivers the FLOPs-vs-latency inversion on GPU |
| `tbl-nn-architectures-dlrm-profile` | Table | ✅ Payoff (L3663) states "too big to fit on a single GPU" and capacity-bound regime |
| `tbl-dl-evolution` | Table | ✅ Cite delivers "each era inherited predecessors' tools + added mechanism for next bottleneck" |
| `tbl-primitive-comparison` | Table | ✅ Cite states the key synthesis: GEMM shared, transformers add content-dependent all-to-all reductions |
| `tbl-sys-design-implications` | Table | ✅ Cite delivers design-checklist function; payoff notes energy-budget connection |
| `tbl-nn-architectures-wildlife-constraints` | Table | ✅ Cite orients the reader and states each constraint forces a specific architectural choice |
| `tbl-nn-architectures-wildlife-risks` | Table | ✅ Cite pairs risks with mitigations explicitly |
| `fig-example-skip-connection` | Figure | ✅ (marginal; cite says "recombination of every building block… illustrated in @fig-example-skip-connection"; payoff at L4113 delivers the RNN→transformer shift with O(S) vs. O(1) metric — enough to pass at Figure level) |

---

## Findings

---

### Finding 1 — `eq-convolution` (Equation / Strict)

**Label:** `eq-convolution` — def L1470
**Verbatim cite:** `The core operation in a CNN can be expressed mathematically as @eq-convolution:`
**Verbatim payoff:** `This equation describes how CNNs process spatial data. $\mathbf{H}^{(\ell)}_{i,j,k}$ is the output at spatial position $(i,j)$ in channel $k$ of layer $\ell$. The triple sum iterates over the filter dimensions: $(m,n)$ scans the spatial filter size, and $c$ covers input channels. $\mathbf{W}^{(\ell)}_{m,n,c,k}$ represents the filter weights, capturing local spatial patterns. Unlike MLPs that connect all inputs to outputs, CNNs only connect local spatial neighborhoods.`

**What is missing:** The payoff paragraph glosses every symbol but stops there. It does not state the equation's system consequence: that this three-nested-sum structure produces an $\mathcal{O}(H_{\text{img}} \times W_{\text{img}} \times K^2 \times C_{\text{in}} \times C_{\text{out}})$ compute cost per layer, that the weight reuse across spatial positions is what creates high arithmetic intensity (compute-bound profile), or that this is the regime contrast with the MLP. A reader who can already read the equation gains nothing from the payoff they could not get from the equation itself; the prose owes the implication.

**Rule:** Equation/Strict — prose must deliver the consequence or regime the equation implies, not merely restate symbol definitions.

**Rule-compliant diff rewrite** (replace the payoff paragraph at L1472):

```diff
- This equation describes how CNNs process spatial data. $\mathbf{H}^{(\ell)}_{i,j,k}$ is the
- output at spatial position $(i,j)$ in channel $k$ of layer $\ell$. The triple sum iterates
- over the filter dimensions: $(m,n)$ scans the spatial filter size, and $c$ covers input
- channels. $\mathbf{W}^{(\ell)}_{m,n,c,k}$ represents the filter weights, capturing local
- spatial patterns. Unlike MLPs that connect all inputs to outputs, CNNs only connect local
- spatial neighborhoods.
+ Three nested summations iterate over the filter's spatial extent $(m,n)$ and its input
+ channels $c$. Each output position $(i,j)$ in feature map $k$ accumulates contributions only
+ from a local $K{\times}K$ neighborhood, so $\mathbf{W}^{(\ell)}_{m,n,c,k}$ is shared
+ identically at every spatial location rather than allocated per position as in an MLP. That
+ sharing is the source of CNN efficiency: the same $K^2 C_{\text{in}}$ weights service every
+ position in the $H_{\text{img}}{\times}W_{\text{img}}$ output grid, producing
+ $\mathcal{O}(H \times W \times K^2 \times C_{\text{in}} \times C_{\text{out}})$ total
+ multiply-accumulates whose reuse ratio makes convolutional layers compute-bound rather than
+ memory-bandwidth-bound.
```

---

### Finding 2 — `eq-rnn` (Equation / Strict)

**Label:** `eq-rnn` — def L2074
**Verbatim cite (this ¶):** `The core operation in a basic RNN can be expressed mathematically as @eq-rnn: … where $\mathbf{h}_t$ denotes the hidden state at time $t$, $\mathbf{x}_t$ denotes the input at time $t$, $\mathbf{W}_{\text{hh}}$ contains the recurrent weights, and $\mathbf{W}_{\text{hx}}$ contains the input weights. Compare the left and right panels of @fig-rnn: the left panel shows the compact recurrent loop, while the right panel unfolds it across time steps, making explicit the temporal dependencies that this recurrence creates.`
**Verbatim payoff (post-float):** `In word sequence processing, each word may be represented as a 100-dimensional vector $(\mathbf{x}_t)$, with a hidden state of 128 dimensions $(\mathbf{h}_t)$. At each time step, the network combines the current input with its previous state to update its sequential understanding, establishing a memory mechanism capable of capturing patterns across time steps.`

**What is missing:** The cite paragraph defines symbols and directs attention to the figure panels but does not state what the equation implies for computation or training. The payoff gives a worked numeric instance of the dimensions but never states the consequence: that $\mathbf{h}_t$ depends on $\mathbf{h}_{t-1}$, creating an $\mathcal{O}(S)$ sequential dependency chain that prevents parallel execution across time steps, or that $\mathbf{W}_{\text{hh}}$ is square (size $d_{\text{hidden}}^2$) which dominates parameter count regardless of sequence length. The system implication of the recurrent structure (the barrier synchronization at each time step, the vanishing-gradient effect from chained products) lives in later prose sections, not as a lead-out from the equation.

**Rule:** Equation/Strict — prose must deliver the consequence or regime the equation implies; a worked instance that only re-reads dimensions is not a consequence.

**Rule-compliant diff rewrite** (replace the payoff paragraph at L2154):

```diff
- In word sequence processing, each word may be represented as a 100-dimensional vector
- $(\mathbf{x}_t)$, with a hidden state of 128 dimensions $(\mathbf{h}_t)$. At each time
- step, the network combines the current input with its previous state to update its
- sequential understanding, establishing a memory mechanism capable of capturing patterns
- across time steps.
+ The critical structural consequence follows directly from the recurrence: $\mathbf{h}_t$
+ cannot be computed before $\mathbf{h}_{t-1}$ is ready, so the $\mathcal{O}(S)$ time steps
+ form a strict sequential dependency chain. No matter how many parallel compute units are
+ available, each step must wait for the previous one to complete. For a 128-dimensional
+ hidden state processing 100 time steps, this means 100 sequential matrix-vector multiplies
+ ($128 \times 128$ recurrent weight matrix $\mathbf{W}_{\text{hh}}$ plus $128 \times 100$
+ input matrix $\mathbf{W}_{\text{hx}}$) with a barrier synchronization between each — the
+ structural reason RNN hardware utilization falls well below compute-bound architectures on
+ the same hardware.
```

---

### Finding 3 — `eq-self-attention` (Equation / Strict)

**Label:** `eq-self-attention` — def L3023
**Verbatim cite:** `The self-attention mechanism differs from earlier attention in one critical respect: every query, key, and value is derived from the same input $\mathbf{X}$, as @eq-self-attention makes explicit:`
**Verbatim payoff:** `Here, $\mathbf{X}$ is the input sequence, and $\mathbf{W}_Q$, $\mathbf{W}_K$, and $\mathbf{W}_V$ are learned weight matrices for queries, keys, and values respectively. This formulation highlights how self-attention derives all its components from the same input, creating a dynamic, content-dependent processing pattern.`

**What is missing:** The payoff repeats the equation's premise (all components from same input) without stating any implication. The equation's key computational consequence — that the $\mathbf{X}\mathbf{W}_Q(\mathbf{X}\mathbf{W}_K)^T$ product is $\mathcal{O}(S^2)$ in the sequence length, or that the softmax normalizes a score matrix that must be fully materialized in memory before the $\mathbf{X}\mathbf{W}_V$ aggregation, or that $d_k$ appears in the denominator to stabilize gradients — is absent from the prose. "Dynamic, content-dependent processing pattern" is a conclusion without the evidence the equation provides.

**Rule:** Equation/Strict — prose must state the consequence or regime (not merely re-describe the structure already visible in the equation).

**Rule-compliant diff rewrite** (replace the payoff paragraph at L3025):

```diff
- Here, $\mathbf{X}$ is the input sequence, and $\mathbf{W}_Q$, $\mathbf{W}_K$, and
- $\mathbf{W}_V$ are learned weight matrices for queries, keys, and values respectively. This
- formulation highlights how self-attention derives all its components from the same input,
- creating a dynamic, content-dependent processing pattern.
+ Here, $\mathbf{X}$ is the input sequence, and $\mathbf{W}_Q$, $\mathbf{W}_K$, and
+ $\mathbf{W}_V$ are learned weight matrices for queries, keys, and values respectively. The
+ inner product $\mathbf{X}\mathbf{W}_Q(\mathbf{X}\mathbf{W}_K)^T$ produces an
+ $S{\times}S$ score matrix: for a 512-token sequence, that is 262,144 scores that must be
+ computed and stored before the softmax aggregates them with $\mathbf{X}\mathbf{W}_V$. This
+ quadratic term in $S$ is the direct cost of allowing every position to attend to every other
+ position simultaneously, and it defines the memory-capacity bottleneck that distinguishes
+ transformers from the constant-memory profile of RNNs.
```

---

### Finding 4 — `eq-multihead-attention` (Equation / Strict)

**Label:** `eq-multihead-attention` — def L3032
**Verbatim cite:** `Each head learns a separate projection into its own subspace, and their outputs are concatenated and linearly mixed, as @eq-multihead-attention formalizes:`
**Verbatim payoff:** `A critical component in both self-attention and multi-head attention is the scaling factor $\sqrt{d_k}$, which serves an important mathematical purpose. This factor prevents the dot products from growing too large, which would push the softmax function into regions with extremely small gradients. For queries and keys of dimension $d_k$, their dot product has variance $d_k$, so dividing by $\sqrt{d_k}$ normalizes the variance to one, maintaining stable gradients and enabling effective learning.`

**What is missing:** The payoff pivots to explain the scaling factor $\sqrt{d_k}$, which belongs to the previously defined attention equation, not to the multi-head formulation itself. The multi-head equation's key implication — that $N_{\text{heads}}$ parallel projections at reduced dimension $d_k = d_{\text{model}}/N_{\text{heads}}$ keep total compute identical to single-head attention while enabling specialization across subspaces — is absent. Neither the parallelism benefit nor the system-level parameter cost ($3 N_{\text{heads}}$ projection matrices per layer) is stated.

**Rule:** Equation/Strict — prose must state the consequence the equation implies; displacing to a nearby equation's explanation does not satisfy this equation's contract.

**Rule-compliant diff rewrite** (replace the payoff paragraph at L3038):

```diff
- A critical component in both self-attention and multi-head attention is the scaling factor
- $\sqrt{d_k}$, which serves an important mathematical purpose. This factor prevents the dot
- products from growing too large, which would push the softmax function into regions with
- extremely small gradients. For queries and keys of dimension $d_k$, their dot product has
- variance $d_k$, so dividing by $\sqrt{d_k}$ normalizes the variance to one, maintaining
- stable gradients and enabling effective learning.
+ The concatenation-and-projection structure has a direct system implication: running
+ $N_{\text{heads}}$ attention heads at reduced dimension $d_k = d_{\text{model}}/N_{\text{heads}}$
+ leaves total compute nearly identical to single-head attention over the full dimension, because
+ each head's $\mathcal{O}(S^2 d_k)$ attention cost sums to $\mathcal{O}(S^2 d_{\text{model}})$
+ across heads. The benefit is representational: each head's independent $\mathbf{W}_i^Q$,
+ $\mathbf{W}_i^K$, $\mathbf{W}_i^V$ projections can specialize for different relationship types
+ (syntax, coreference, proximity), while the final $\mathbf{W}^O$ projection recombines them
+ into a shared representation. The scaling factor $\sqrt{d_k}$ in the underlying attention
+ computation prevents dot products from growing with $d_k$, keeping the softmax gradient in a
+ stable regime regardless of how many heads the dimension is split across.
```

---

### Finding 5 — `fig-mlp` (Figure / High)

**Label:** `fig-mlp` — def L847
**Verbatim cite:** `Dense connectivity translates directly into fully connected layers and matrix multiplication operations, the mathematical basis introduced in @sec-neural-computation-matrix-multiplication-formulation-417c that makes MLPs computationally tractable. @Fig-mlp shows how each layer transforms its input through this core operation.`
**Payoff paragraph (after float):** The scanner's "payoff ¶" jumps far forward to the `eq-dense-layer` citation paragraph (L996). No body prose immediately follows the figure.

**What is missing:** The cite sentence is a pure float-announcer ("shows how each layer transforms its input through this core operation"). There is no lead-out in the surrounding body prose telling the reader what the dense all-to-all connectivity pattern implies: that every output neuron requires contributions from every input neuron, producing $\mathcal{O}(N \times M)$ multiply-accumulate operations per layer, and that this is the structural reason MLPs are memory-bandwidth-bound (weights must be loaded once per sample with no spatial reuse). The takeaway lives only in the caption.

**Rule:** Figure/High — prose must tell the figure's story; "shows how" with no follow-through is the canonical failing pattern.

**Rule-compliant diff rewrite** (replace cite sentence and add lead-out immediately before the figure):

```diff
- Dense connectivity translates directly into fully connected layers and matrix multiplication
- operations, the mathematical basis introduced in
- @sec-neural-computation-matrix-multiplication-formulation-417c that makes MLPs
- computationally tractable. @Fig-mlp shows how each layer transforms its input through this
- core operation.
+ Dense connectivity translates directly into fully connected layers and matrix multiplication
+ operations, the mathematical basis introduced in
+ @sec-neural-computation-matrix-multiplication-formulation-417c that makes MLPs
+ computationally tractable. @Fig-mlp illustrates the consequence of all-to-all connectivity:
+ every neuron in one layer connects to every neuron in the next, so for a 784-input to
+ 100-hidden-unit layer, each output requires 784 multiply-accumulate operations and the full
+ $784{\times}100$ weight matrix must be loaded from memory for every input sample. Unlike
+ convolutional layers that reuse the same weights across spatial positions, dense layers load
+ each weight exactly once per sample, making memory bandwidth — not arithmetic throughput —
+ the binding constraint.
```

---

### Finding 6 — `fig-cnn-spatial-processing` (Figure / High)

**Label:** `fig-cnn-spatial-processing` — def L1237
**Verbatim cite (L1455):** `As @fig-cnn-spatial-processing illustrates, convolutional neural networks meet both requirements through hierarchical feature extraction, where simple patterns compose into increasingly complex representations at successive layers.`
**Note on position:** The figure definition is at L1237, the body text that sets it up (L1233–1235) appears *before* the figure, and the only cite (L1455) is also *before* the figure in a section added after the figure's TikZ block ends (L1451–1453 is the payoff prose but it precedes the cite at L1455).

**What is missing:** The sole cite is the classic float-announcer: "As @fig-cnn-spatial-processing illustrates, CNNs meet both requirements through hierarchical feature extraction." The prose tells the reader what the figure shows but delivers no specific interpretation of what the figure demonstrates — there is no statement of what the hierarchical stacking from local edges to global objects implies for system design (weight sharing, translation equivariance, reduced parameter count). The payoff prose at L1453 runs before the figure and sets up the context; after the figure's TikZ block closes at L1451, the next prose (L1453) is reused as the "payoff" but structurally it precedes the cite. No prose after the figure-closing `:::` delivers a takeaway.

**Rule:** Figure/High — content leads, reference rides along; "as @fig illustrates, X does Y" with no follow-through is the failing pattern.

**Rule-compliant diff rewrite** (replace cite sentence at L1455):

```diff
- As @fig-cnn-spatial-processing illustrates, convolutional neural networks meet both
- requirements through hierarchical feature extraction, where simple patterns compose into
- increasingly complex representations at successive layers.
+ Convolutional neural networks meet both requirements through the hierarchical feature
+ extraction shown in @fig-cnn-spatial-processing: early filters respond to low-level
+ edges and textures, and successive layers combine those responses into progressively
+ larger receptive fields until the final layer integrates global object structure. The
+ system consequence of this hierarchy is that most parameters reside in the deep layers
+ processing abstract features, while the spatial resolution — and thus the memory footprint
+ for activations — shrinks at each pooling stage. A network detecting a cat uses the same
+ edge detectors everywhere in the image; the spatial hierarchy, not a unique detector per
+ location, is what keeps parameter count tractable.
```

---

### Finding 7 — `fig-cnn` (Figure / High)

**Label:** `fig-cnn` — def L1482
**Verbatim cite (L1480):** `Study the mechanics in @fig-cnn: a small filter slides over the input image, computing a dot product at each position to generate a feature map. This sliding window captures local structures while maintaining translation equivariance—the same filter detects the same pattern regardless of where it appears.`
**Payoff paragraph:** The scanner resolves the payoff paragraph as L1551, which opens a new numerical example about applying a CNN to MNIST images. No paragraph immediately follows the figure's `:::` close.

**What is missing:** The cite is a narrated directive ("Study the mechanics in @fig-cnn: a small filter slides…") followed by a statement that also appears in the caption ("the same filter detects the same pattern regardless of where it appears"). The lead-out that the prose owes — what the sliding-window mechanics imply about the compute cost (each $3{\times}3$ filter position requires 9 MACs, parameter sharing reduces weights by ~5,600× for a $224{\times}224$ input compared to a dense MLP) — lives only in the caption, not in body prose. There is no prose body paragraph after the figure that interprets it.

**Rule:** Figure/High — the caption quantifying parameter reduction (5,600×, 9 MACs per position) does not count toward the prose contract.

**Rule-compliant diff rewrite** (add a lead-out paragraph after the figure's `:::` close at ~L1490):

```diff
+ The sliding-window mechanics in @fig-cnn expose what makes convolution computationally
+ attractive: the same $3{\times}3$ filter is applied at every spatial position, so the
+ 27 weights (nine per input channel) serve the entire $H_{\text{img}}{\times}W_{\text{img}}$
+ output grid. For a $224{\times}224$ RGB image this means 27 parameters replace the
+ 150,528-weight input connection of an equivalent dense layer — roughly a 5,600$\times$
+ reduction. The cost paid is structured compute: $H \times W$ dot products must be computed,
+ one per spatial position, but because those dot products are independent they map naturally
+ to parallel hardware. The result is a layer whose compute cost scales with image area but
+ whose parameter count depends only on filter size and channel counts, creating the
+ arithmetic intensity profile that makes CNNs compute-bound under batch execution.
```

---

### Finding 8 — `fig-rnn` (Figure / High)

**Label:** `fig-rnn` — def L2077
**Verbatim cite:** `Compare the left and right panels of @fig-rnn: the left panel shows the compact recurrent loop, while the right panel unfolds it across time steps, making explicit the temporal dependencies that this recurrence creates.`
**Verbatim payoff (L2154):** `In word sequence processing, each word may be represented as a 100-dimensional vector $(\mathbf{x}_t)$, with a hidden state of 128 dimensions $(\mathbf{h}_t)$. At each time step, the network combines the current input with its previous state to update its sequential understanding, establishing a memory mechanism capable of capturing patterns across time steps.`

**What is missing:** The cite describes panel contents ("compact recurrent loop… unfolds across time steps") but does not state the system insight the unfolded view delivers. The payoff gives a dimension example without stating what the unfolded view shows about weight sharing: three matrices ($\mathbf{W}_{\text{hh}}$, $\mathbf{W}_{\text{hx}}$, $\mathbf{W}_{\text{yh}}$) repeat identically across every column of the unfolded diagram, which is the visual proof that parameter count is $\mathcal{O}(d_{\text{hidden}}^2)$ regardless of sequence length. That insight lives only in the caption.

**Rule:** Figure/High — the caption's weight-sharing insight (constant $\mathcal{O}(d_{\text{hidden}}^2)$ parameter count) does not count toward the prose contract.

**Rule-compliant diff rewrite** (replace the payoff paragraph at L2154):

```diff
- In word sequence processing, each word may be represented as a 100-dimensional vector
- $(\mathbf{x}_t)$, with a hidden state of 128 dimensions $(\mathbf{h}_t)$. At each time
- step, the network combines the current input with its previous state to update its
- sequential understanding, establishing a memory mechanism capable of capturing patterns
- across time steps.
+ The unfolded view in @fig-rnn makes the weight-sharing property visible: the same three
+ matrices — $\mathbf{W}_{\text{hx}}$, $\mathbf{W}_{\text{hh}}$, and $\mathbf{W}_{\text{yh}}$
+ — are labeled identically at every time step. Because all $S$ steps share one parameter set,
+ the parameter count is $\mathcal{O}(d_{\text{hidden}}^2)$ independent of sequence length,
+ giving RNNs a memory footprint that does not grow with the sequences they process. The
+ vertical arrows between columns show the dependency chain: each hidden state feeds the next,
+ so a sequence of length 100 requires 100 sequential matrix-vector multiplies before any
+ output can be produced — the structural bottleneck that makes RNN hardware utilization
+ sensitive to sequence length rather than batch size.
```

---

### Finding 9 — `fig-attention` (Figure / High)

**Label:** `fig-attention` — def L2418
**Verbatim cite (L2416):** `The attention operation involves several key steps. First, it computes query, key, and value projections for each position in the sequence. Next, examine the $S{\times}S$ attention matrix in @fig-attention — each cell represents a query-key interaction, and the color intensity reveals which positions attend most strongly to which others. Finally, these attention weights combine value vectors to produce the output.`
**Payoff paragraph:** Scanner resolves payoff as L2538 (a cite sentence for `@fig-attention-weightcalc`). No prose immediately follows the figure's close.

**What is missing:** The cite describes what to look at (S×S cells, color intensity) but does not state what the matrix structure implies: that storing and computing this $S^2$ matrix is the $\mathcal{O}(S^2)$ memory bottleneck, or that the 36 interactions shown for a 6-token sequence scale quadratically to 262,144 for a 512-token sequence. The prose goes from describing the figure to immediately citing the next figure; the "so what" of attention's $S^2$ cost lives only in the caption.

**Rule:** Figure/High — the caption's "thirty-six similarity computations… memory bottleneck: storing $S^2$ attention weights" does not count toward the prose contract.

**Rule-compliant diff rewrite** (add a lead-out sentence after the three-step description, before the figure):

```diff
- The attention operation involves several key steps. First, it computes query, key, and value
- projections for each position in the sequence. Next, examine the $S{\times}S$ attention
- matrix in @fig-attention — each cell represents a query-key interaction, and the color
- intensity reveals which positions attend most strongly to which others. Finally, these
- attention weights combine value vectors to produce the output.
+ The attention operation involves several key steps. First, it computes query, key, and value
+ projections for each position in the sequence. Then, the $S{\times}S$ attention matrix in
+ @fig-attention collects a score for every query-key pair: for the 6-token example, 36 cells;
+ for a 512-token sequence, 262,144 cells that must all be computed and held in memory before
+ the softmax normalizes them. Finally, these attention weights combine value vectors to
+ produce the output. The quadratic growth of that score matrix — not the projection math —
+ is what makes long-sequence attention memory-capacity-bound, and why engineering effort has
+ concentrated on approximating or replacing it rather than optimizing the linear projection
+ steps.
```

---

### Finding 10 — `fig-attention-weightcalc` (Figure / High)

**Label:** `fig-attention-weightcalc` — def L2579
**Verbatim cite (L2538):** `Unlike the fixed weight matrices found in previous architectures, attention weights are computed dynamically for each input. Follow the matrix dimensions in @fig-attention-weightcalc to see this dynamic computation unfold: the embedding matrix multiplies with QKV weight matrices in a single batched operation, and the resulting projections change for every new input sequence.`
**Payoff paragraph:** Scanner resolves payoff as L2737 (a cite sentence for `@lst-attention_layer_compute`, a different float). No prose immediately follows the figure's close.

**What is missing:** The cite is a directive ("Follow the matrix dimensions") that narrates what the figure shows but does not state the consequence. The "dynamic" observation recapitulates the preceding paragraph without adding a takeaway. The system implication — that the single batched $6 \times 768 \times 2304$ operation (combining all three QKV projections into one GEMM) is three times more hardware-efficient than three sequential projections because it keeps the weight matrix resident in the hardware's register file across all three outputs — lives only in the caption.

**Rule:** Figure/High — the caption's batched-GEMM efficiency rationale does not count toward the prose contract.

**Rule-compliant diff rewrite** (replace cite sentence at L2538):

```diff
- Unlike the fixed weight matrices found in previous architectures, attention weights are
- computed dynamically for each input. Follow the matrix dimensions in
- @fig-attention-weightcalc to see this dynamic computation unfold: the embedding matrix
- multiplies with QKV weight matrices in a single batched operation, and the resulting
- projections change for every new input sequence.
+ Unlike the fixed weight matrices found in previous architectures, attention weights are
+ computed dynamically for each input. @Fig-attention-weightcalc shows why frameworks fuse
+ the three projections into one operation: rather than executing three separate
+ $6{\times}768{\times}768$ matrix multiplications, a single $6{\times}768{\times}2304$
+ GEMM produces all query, key, and value projections in one pass. For GPT-2 with a
+ 768-dimensional model, this single batched multiply requires approximately
+ `{python} QKVProjectionCosts.qkv_macs_str` — the same total work, but with the weight
+ matrix loaded from HBM once rather than three times, cutting memory traffic by a factor
+ of three and enabling higher hardware utilization.
```

---

### Finding 11 — `lst-rnn_layer_step` (Listing / Medium)

**Label:** `lst-rnn_layer_step` — def L2216
**Verbatim cite (L2230):** `@Lst-rnn_layer_step demonstrates the operation using high-level matrix operations found in deep learning frameworks. The function handles a single time step, taking the current input \`x_t\` and previous hidden state \`h_prev\`, along with two weight matrices: \`W_hh\` for hidden-to-hidden connections and \`W_hx\` for input-to-hidden connections. Through matrix multiplication operations (\`matmul\`), it merges the previous state and current input to generate the next hidden state.`
**Payoff paragraph (L2230):** Same sentence as the cite (the payoff paragraph is the cite itself). L2232 provides the follow-up.

**What is missing:** The cite describes what the code does (handles one time step, merges state and input) but does not identify the mechanism to look for or the design choice that matters. The listing-level standard requires the prose to say what the code *shows* — the mechanism it embodies — and what the reader should notice. Here the reader should notice that `tanh(matmul(h_prev, W_hh) + matmul(x_t, W_hx) + b)` is two GEMVs per time step (not one, because they are vector-matrix multiplies rather than matrix-matrix), which is the structural reason RNNs cannot saturate hardware designed for GEMM. That insight appears in the next paragraph (L2232) but is not anchored to what the listing shows.

**Rule:** Listing/Medium — prose must state the mechanism the listing embodies and what to notice; "demonstrates the operation" with a paraphrase of function parameters is not sufficient framing.

**Rule-compliant diff rewrite** (replace cite paragraph at L2230):

```diff
- @Lst-rnn_layer_step demonstrates the operation using high-level matrix operations found in
- deep learning frameworks. The function handles a single time step, taking the current input
- `x_t` and previous hidden state `h_prev`, along with two weight matrices: `W_hh` for
- hidden-to-hidden connections and `W_hx` for input-to-hidden connections. Through matrix
- multiplication operations (`matmul`), it merges the previous state and current input to
- generate the next hidden state.
+ @Lst-rnn_layer_step exposes the per-step cost of the recurrence: two separate `matmul`
+ calls, one for the hidden-to-hidden term (`h_prev @ W_hh`) and one for the input term
+ (`x_t @ W_hx`), followed by a bias addition and activation. Each call is a matrix-vector
+ multiply (GEMV), not a matrix-matrix multiply (GEMM), because the hidden state is a
+ single vector per sequence in the batch. Hardware accelerators optimized for GEMM deliver
+ orders-of-magnitude higher throughput for GEMM than GEMV; the two-GEMV structure is the
+ arithmetic reason RNNs fail to saturate accelerators even when batch size is large.
```

---

### Finding 12 — `tbl-architecture-families` (Table / High)

**Label:** `tbl-architecture-families` — def L150
**Verbatim cite (L136):** `Five architectural families define neural computation, each optimized for different data characteristics. @Tbl-architecture-families maps each family to its data domain, core innovation, and dominant system bottleneck.`
**Verbatim post-table prose (L138):** `Each architectural choice creates distinct computational signatures that propagate through every level of the implementation stack.`

**What is missing:** The cite names three columns and states the mapping, but reads the table back to the reader ("maps each family to its data domain, core innovation, and dominant system bottleneck") rather than stating what the mapping reveals. The post-table prose (L138) says "distinct computational signatures propagate" — this is a near-truism that could follow any architecture table. The table's actual load-bearing insight — that the bottleneck column reads off the iron law (dense activations → memory bandwidth; spatial reuse → compute throughput; sequential dependencies → parallelism loss; quadratic attention → memory capacity) — lives exclusively in the caption, not the body prose.

**Rule:** Table/High — prose must deliver the conclusion the table encodes, not merely re-state what the columns contain.

**Rule-compliant diff rewrite** (replace cite and post-table prose at L136–138):

```diff
- Five architectural families define neural computation, each optimized for different data
- characteristics. @Tbl-architecture-families maps each family to its data domain, core
- innovation, and dominant system bottleneck.
-
- Each architectural choice creates distinct computational signatures that propagate through
- every level of the implementation stack.
+ Five architectural families define neural computation, each exposing a different
+ bottleneck that follows directly from the iron law. @Tbl-architecture-families makes this
+ mapping concrete. MLPs move dense weight matrices through memory once per sample with no
+ reuse, so bandwidth — not arithmetic throughput — is the binding constraint. CNNs reuse
+ the same small filter across every spatial position, producing high arithmetic intensity
+ and shifting the bottleneck to compute throughput. RNNs chain hidden-state updates
+ sequentially, so parallelism across time steps is structurally impossible regardless of
+ hardware capacity. Transformers materialize an $S{\times}S$ attention score matrix,
+ making memory capacity scale quadratically with sequence length. DLRM's embedding tables
+ can reach terabytes, exhausting device memory before any arithmetic begins. Every
+ architectural choice is a bottleneck selection, and that selection propagates through every
+ level of the implementation stack.
```

---

### Finding 13 — `tbl-nn-architectures-gpt2-profile` (Table / High)

**Label:** `tbl-nn-architectures-gpt2-profile` — def L3310
**Verbatim cite (L3300):** `…@Tbl-nn-architectures-gpt2-profile summarizes the bandwidth lighthouse's quantitative properties:`
**Verbatim payoff (L3374):** Discusses KV cache linear growth — a downstream consequence not directly interpreting the profile table rows.

**What is missing:** The cite is a bare "summarizes" pointer. Unlike the ResNet-50 and MobileNetV2 profiles (whose payoffs at L1971 and L1998 deliver specific cross-hardware insight), the GPT-2 profile payoff paragraph discusses the KV cache, which is a forward pointer to inference mechanics rather than an interpretation of the profile table's specific rows (parameters, model size, compute, constraint, profile). The table's key finding — that loading `{python} GPT2BandwidthProfile.gpt2_fp32_gb_str` of weights per generated token at ~0.5 FLOP/byte leaves compute cores idle while memory bandwidth saturates — is stated in the surrounding lighthouse callout prose but not as a direct payoff from the table's cells.

**Rule:** Table/High — "summarizes the quantitative properties" is the canonical failing form; the H&P standard requires a "key result is…" sentence that the reader could use to skip the cells.

**Rule-compliant diff rewrite** (add a lead-out sentence after the table's closing row at ~L3308):

```diff
+ The key result in @tbl-nn-architectures-gpt2-profile is in the Compute row: each generated
+ token triggers only a single matrix-vector multiply per layer, delivering roughly
+ `{python} GPT2BandwidthProfile.gpt2_gflop_per_token_str` of useful arithmetic against
+ `{python} GPT2BandwidthProfile.gpt2_fp32_gb_str` of weight data loaded from HBM — an
+ arithmetic intensity of about `{python} GPT2BandwidthProfile.intensity_fp32_str`. Modern
+ accelerators need intensities in the hundreds of FLOP/byte to saturate their compute units,
+ so GPT-2 XL inference leaves those units almost entirely idle, and generation throughput
+ tracks HBM bandwidth linearly rather than FLOP/s.
```

---

### Finding 14 — `tbl-normalization-comparison` (Table / High)

**Label:** `tbl-normalization-comparison` — def L4028
**Verbatim cite (L4018):** `The choice between normalization variants depends on computational context. @Tbl-normalization-comparison summarizes the key trade-offs. BatchNorm typically stores learned scale/shift parameters plus nonlearned running mean/variance buffers; LayerNorm computes per-sample statistics at runtime and typically stores learned scale/shift parameters but no running-statistic buffers.`
**Verbatim payoff (L4030):** `Batch size constraints emerge because batch normalization requires sufficiently large batches for stable statistics. Empirically, batch sizes below 16 degrade performance noticeably, and sizes below 8 can cause training instability. This constraint impacts memory-limited scenarios such as high-resolution images or billion-parameter models.`

**What is missing:** The cite says "summarizes the key trade-offs" and then describes storage internals (what each variant stores). The payoff delivers the batch-size constraint for BatchNorm. However, the table's load-bearing contrast — the decision criterion that determines *which variant to choose* — is never stated in prose. The Training/Inference row (BatchNorm uses running statistics at inference; LayerNorm and RMSNorm behave identically at train and inference) is the operationally critical difference for deployment, but body prose never calls this out. The reader must read the cells to learn why transformers universally prefer LayerNorm for autoregressive inference.

**Rule:** Table/High — the decision the table drives must be stated in body prose, not left to the reader to extract from the cells.

**Rule-compliant diff rewrite** (add a lead-out sentence after the payoff paragraph at L4030):

```diff
+ The Training/Inference row in @tbl-normalization-comparison carries the deployment decision:
+ BatchNorm requires running-statistic buffers that are estimated during training and frozen at
+ inference, making its behavior differ between the two phases and creating correctness risks
+ when inference batch sizes differ from training. LayerNorm and RMSNorm compute statistics
+ fresh from each sample at both train and inference time, so their behavior is identical in
+ both phases. This is why transformers deployed for autoregressive generation — where the
+ "batch" at inference is typically a single token — universally prefer LayerNorm or RMSNorm
+ over BatchNorm.
```

---

### Finding 15 — `tbl-arch-complexity` (Table / High)

**Label:** `tbl-arch-complexity` — def L4306
**Verbatim cite (first, L4297):** `@Tbl-arch-complexity quantifies how these different memory access patterns contribute to the overall memory requirements of each architecture, comparing MLPs, CNNs, RNNs, and transformers across parameter storage, activation storage, and scaling behavior.`
**Verbatim post-table prose (L4322):** `@Tbl-arch-complexity captures *where* data lives and how access patterns scale. The complementary @tbl-computational-complexity that follows captures *how much* computation each architecture demands…`

**What is missing:** The first cite says "quantifies how memory access patterns contribute" — another bare pointer. The second cite (L4322) frames the pairing with the complexity table but still does not deliver the table's key finding. The table's load-bearing row is the Transformer row: $\mathcal{O}(B \times S^2)$ activation storage for attention, which is the reason long-context transformers require activation checkpointing and why context length is a hard memory constraint. Neither cite states this. The payoff sentence at L4308 is "Where:" followed by a notation legend (not a takeaway).

**Rule:** Table/High — "quantifies how X contributes" is the canonical bare-pointer form; the specific finding (which row matters, which regime is surprising) must appear in prose.

**Rule-compliant diff rewrite** (replace payoff at L4308, adding a real takeaway before the notation legend):

```diff
- Where:
+ The critical row is the Transformer: attention activations scale as $\mathcal{O}(B \times
+ S^2)$, so doubling the context window quadruples the activation memory required to hold
+ the intermediate attention matrices during a forward pass. At sequence length 4,096 with
+ batch size 8, that is roughly 128 million scalar values — dwarfing the parameter storage
+ and explaining why long-context training requires activation checkpointing (recomputing
+ activations during the backward pass rather than storing them). By contrast, RNNs maintain
+ only an $\mathcal{O}(d_{\text{hidden}})$ hidden state during inference regardless of
+ sequence length, giving them a memory footprint for inference that is independent of context
+ size.
+
+ Where:
```

---

### Finding 16 — `tbl-computational-complexity` (Table / High)

**Label:** `tbl-computational-complexity` — def L4335
**Verbatim cite (L4322):** `The complementary @tbl-computational-complexity that follows captures *how much* computation each architecture demands, including forward-pass FLOPs, parallelization potential, and the resulting bottleneck. Together, the two tables answer the systems questions "how much work?" and "how does the memory system handle it?", providing a resource profile that informs design decisions such as choosing memory hierarchy configurations and developing memory optimization strategies.`
**Verbatim payoff (L4337):** `Understanding these memory access patterns is essential as architectures evolve. The shift from CNNs to transformers, for instance, has driven the development of hardware with larger on-chip memories and more advanced caching strategies to handle increased working sets and more dynamic access patterns. Future architectures will likely continue to be shaped by their memory access characteristics as much as their computational requirements.`

**What is missing:** The cite gives a structural orientation ("answers 'how much work?' and 'how does the memory system handle it?'") but does not state the table's specific finding. The payoff discusses hardware evolution trends but never names the specific row or contrast that the table encodes. The table's decision-driving insight is in the Parallelization column: RNNs alone show "Poor (Sequential deps)" while all other architectures show "High" or "Excellent" — this is the quantitative proof that RNNs cannot use accelerator parallelism, not just an assertion. Neither cite nor payoff directs the reader to this column.

**Rule:** Table/High — the load-bearing contrast (the specific row or column that drives the point) must be named in prose.

**Rule-compliant diff rewrite** (replace payoff at L4337):

```diff
- Understanding these memory access patterns is essential as architectures evolve. The shift
- from CNNs to transformers, for instance, has driven the development of hardware with larger
- on-chip memories and more advanced caching strategies to handle increased working sets and
- more dynamic access patterns. Future architectures will likely continue to be shaped by
- their memory access characteristics as much as their computational requirements.
+ The Parallelization column in @tbl-computational-complexity makes the most consequential
+ contrast explicit: RNNs alone read "Poor (Sequential deps)" while every other architecture
+ achieves high or excellent position-level parallelism. This single distinction explains the
+ industry transition from RNNs to transformers: not better accuracy on short sequences, but
+ the ability to keep all available accelerator cores busy during both training and inference.
+ The Bottleneck column reinforces this: where MLPs and CNNs are limited by memory bandwidth
+ or compute throughput — resources that can be expanded — RNNs are limited by sequential
+ dependencies that are intrinsic to the architecture and cannot be overcome with faster or
+ more hardware.
```

---

*End of worklist — 16 findings, 0 fails, all partial (⚠️). Floats not listed above pass the standard.*
