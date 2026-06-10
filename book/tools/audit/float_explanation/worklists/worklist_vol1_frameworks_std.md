# Float Exposition Worklist — `frameworks.qmd` (vol1)

Evaluated against `FLOAT_EXPOSITION_STANDARD.md`. Caption, alt-text, in-figure labels,
and code comments do NOT count toward the prose's job. Only running body prose counts.

---

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:------|-------:|---:|---:|---:|
| Algorithm | 🔴 strict | 1 | 1 | 0 | 0 |
| Equation | 🔴 strict | 4 | 3 | 1 | 0 |
| Figure | 🟠 high | 14 | 14 | 0 | 0 |
| Listing | 🟡 medium | 32 | 22 | 10 | 0 |
| Table | 🟠 high | 17 | 13 | 4 | 0 |
| **Total** | | **68** | **53** | **15** | **0** |

---

## Findings

All findings are ⚠️ Partial. No 🛑 Fails (no orphans, no bare-pointer-only floats).

---

### F-01 · `eq-compilation-benefit` — Equation 🔴

**Verbatim cite (L1770):**
> "The optimal compilation strategy depends on the ratio of *development iterations* to *production executions*, defined in @eq-compilation-benefit:"

**What is missing:** The prose delivers the symbol-naming "Where:" list immediately after (L1776–1782) and a decision rule one line later (L1784: "compile when Compilation Benefit > 1"). However, no sentence states the *regime implication* in words: what happens when $N_{\text{dev}} \gg N_{\text{prod}}$, or when $T_{\text{compile}} \to 0$. The equation's structural insight — that benefit grows linearly with production executions and shrinks as development iterations increase — is never stated in prose near the equation. The implication surfaces only paragraphs later at L1886, leaving the equation without an immediate "so what." Passes the symbol-naming test; fails the implication test at the strict equation level.

**Missing move:** Add one sentence immediately after the "Where:" list that states the asymptotic behavior (or at minimum the regime signal: high $N_{\text{prod}}$, low $N_{\text{dev}}$, cheap $T_{\text{eager}} - T_{\text{compiled}}$ each pulls benefit in opposite directions).

**Rule-compliant diff rewrite** (insert after L1784, replacing the current bare decision-rule line):

```diff
- The decision rule is to compile when $\text{Compilation Benefit} > 1$. The ratio is dimensionless.
+ The decision rule is to compile when $\text{Compilation Benefit} > 1$. The ratio is dimensionless. Two regimes
+ dominate in practice: when $N_{\text{prod}} \gg N_{\text{dev}}$ (long training runs or sustained inference),
+ the numerator grows without bound and compilation always pays; when $N_{\text{dev}} \gg N_{\text{prod}}$
+ (rapid prototyping with frequent architecture changes), the denominator dominates and compilation costs
+ more than it saves. The three subsequent sections quantify each regime against real throughput numbers.
```

---

### F-02 · `tbl-tracing-vs-scripting` — Table 🟠

**Verbatim cite (L1398, post-float backward pointer):**
> "@Tbl-tracing-vs-scripting summarizes the key trade-offs between these two approaches."

**What is missing:** Two failures. First, there is no forward lead-in before the table — the table appears inline at L1387 immediately after a scripting-constraints paragraph, with no cross-reference introducing it before it appears. The only cite is a backward pointer at L1398 after the reader has already seen the table. Second, that post-table sentence "summarizes the key trade-offs" is a pointer-without-takeaway: it names the table's subject, not what the reader should conclude from it. The actionable insight (when to choose tracing vs. scripting, and *why* the silent-failure risk makes the distinction system-critical) lives only in the caption and in the table's "Best for" row, not in body prose.

**Missing move:** Add a pre-table sentence citing the float and naming the decision, then add one post-table sentence replacing "summarizes" with the actual takeaway.

**Rule-compliant diff rewrite:**

```diff
- These constraints are the price of compilation: every feature that makes Python flexible also makes it
- unpredictable for a compiler.
+ These constraints are the price of compilation: every feature that makes Python flexible also makes it
+ unpredictable for a compiler. The choice between the two strategies reduces to one question: does the
+ model's control flow depend on tensor values at runtime? @Tbl-tracing-vs-scripting maps each approach
+ to the class of models it can correctly handle.
```

And replace the post-table pointer:

```diff
- @Tbl-tracing-vs-scripting summarizes the key trade-offs between these two approaches.
+ The table's central lesson is correctness before performance: tracing's silent failure mode — baking in
+ one branch and producing wrong results for the other — is the reason scripting exists, not merely style.
+ @Tbl-tracing-vs-scripting maps both approaches to the model classes they can correctly handle.
```

---

### F-03 · `tbl-mlfm-graphs` — Table 🟠

**Verbatim cite (L1743):**
> "Beyond these core execution trade-offs, @tbl-mlfm-graphs highlights additional systems-level distinctions between static and dynamic approaches."

**Payoff (L1754):**
> "These trade-offs are not binary choices. Modern frameworks offer a spectrum of options, which raises the quantitative question of where on this spectrum a given project should operate."

**What is missing:** The citation sentence is a pointer ("highlights additional distinctions"), not an interpretation. The payoff pivots away to the next topic without extracting the table's specific finding. The key result in the cells — that static graphs' advantage in memory management and hardware utilization comes at the cost of research velocity and legacy-code integration — is never stated in body prose. A reader who skips the table cells cannot infer the decision boundary from the surrounding prose.

**Missing move:** Replace or augment the payoff sentence to name the load-bearing contrast: static graphs pay research-velocity and integration cost to gain memory efficiency and hardware-specific code generation.

**Rule-compliant diff rewrite:**

```diff
- These trade-offs are not binary choices. Modern frameworks offer a spectrum of options, which raises
- the quantitative question of where on this spectrum a given project should operate.
+ The table's organizing contrast is memory and utilization against research velocity. Static graphs
+ buy precise allocation planning and hardware-specific code generation at the cost of the slower
+ define-then-run cycle that reduces research velocity and complicates integration with imperative code.
+ This trade-off is not binary — modern frameworks offer a spectrum of options — which raises the
+ quantitative question of where on this spectrum a given project should operate.
```

---

### F-04 · `tbl-training-benchmark` — Table 🟠

**Verbatim primary cite (L1856):**
> "@Tbl-training-benchmark provides representative throughput data across execution modes and model architectures:"

**Payoff (L1867):**
> "These throughput differences across execution modes raise a practical question — which framework execution strategy best serves each workload archetype."

**What is missing:** The first cite is a bare pointer ("provides data"). The payoff pivots to a question rather than delivering the table's specific conclusion. The load-bearing result — that torch.compile provides 1.4–1.5× speedup while TensorRT provides 2.3–2.6× but is inference-only, and that compile time scales with model complexity such that ResNet-50 (fast model, 15–30 s) has a different breakeven than GPT-2 (slow model, 45–90 s) — lives only in the cells. The second use at L1882 does make a correct specific claim about ResNet-50 and TensorRT speedup, but that is a downstream reference rather than an immediate lead-out from the first citation.

**Missing move:** The payoff sentence should extract the table's central finding before pivoting.

**Rule-compliant diff rewrite:**

```diff
- These throughput differences across execution modes raise a practical question — which framework
- execution strategy best serves each workload archetype.
+ Across all four models, torch.compile delivers consistent 1.4–1.5× throughput gains over eager mode,
+ while TensorRT delivers 2.3–2.6× gains at the cost of longer compilation and inference-only coverage.
+ The compile-time column reveals a second constraint: GPT-2's 45–90-second compile window versus
+ ResNet-50's 15–30-second window means the breakeven point shifts with model size. These throughput
+ differences raise the practical question of which execution strategy best serves each workload archetype.
```

---

### F-05 · `tbl-nsight-metrics` — Table 🟠

**Verbatim cite (L3455):**
> "@Tbl-nsight-metrics lists the key metrics to examine when optimizing ML kernels."

**Payoff:** Immediately after the table a new section heading appears (L3466 "Data pipelines and loading"). There is no lead-out paragraph for this table. The scanner's "payoff" entry (L3527) is from a different section and does not reference the table.

**What is missing:** The citation sentence is a pointer only ("lists the key metrics"). There is no body prose stating what a reader should conclude from the table — specifically, what the four metrics tell you collectively (SM Occupancy for parallelism, Memory and Compute Throughput together identify the roofline bound, Tensor Core Active confirms precision utilization). The decision logic (low SM Occupancy → improve parallelism; low Memory Throughput → fix access patterns; low Compute Throughput → reduce memory bottleneck) lives only in the "Optimization Target" column, not in prose.

**Missing move:** Add a one- to two-sentence lead-out after the table before the section break that states the collective diagnostic pattern.

**Rule-compliant diff rewrite** (insert between L3464 and L3466):

```diff
+ These four metrics work as a diagnostic sequence: SM Occupancy identifies whether parallelism is
+ saturated; Memory and Compute Throughput together locate the roofline bound (if Memory Throughput is
+ near peak but Compute Throughput is low, the kernel is memory-bound, not compute-deficient); and Tensor
+ Core Active confirms that a mixed-precision pass actually exercises the specialized units rather than
+ falling back to FP32. A kernel with high Compute Throughput but low Tensor Core Active is leaving
+ mixed-precision speedup on the table.
+
```

---

### F-06 · `lst-torchscript-ir` — Listing 🟡

**Verbatim cite (L1363):**
> "To understand what the compiler produces, @lst-torchscript-ir inspects the generated intermediate representation directly."

**Payoff (L1385):**
> "However, scripting imposes constraints on what Python constructs are supported…"

**What is missing:** The cite names the purpose ("understand what the compiler produces") but identifies no mechanism or design choice for the reader to notice. The IR listing reveals that scalar multiplication `x * 2` becomes `prim::Constant[value=2]` + `aten::mul` — two separate primitive nodes — demonstrating why the IR is a lower-level, hardware-agnostic representation suitable for optimization passes. The payoff paragraph pivots entirely to scripting constraints and never names what the IR structure shows. A reader who skips the listing learns nothing about what TorchScript IR looks like or why that representation matters.

**Missing move:** Name the key observation in the lead-in (what to look at in the IR output) and add a brief lead-out connecting the IR structure to the optimization property.

**Rule-compliant diff rewrite:**

```diff
- To understand what the compiler produces, @lst-torchscript-ir inspects the generated intermediate
- representation directly.
+ To understand what the compiler produces, @lst-torchscript-ir prints the TorchScript IR for a
+ simple two-operation function. The output reveals the key property: Python's `x * 2 + 1` becomes
+ two separate `aten::` (tensor operation) nodes and two `prim::Constant` nodes — a flat, typed,
+ hardware-agnostic graph that optimization passes can rewrite without touching Python.
```

And after the float, before L1385:

```diff
+ The IR's `prim::` and `aten::` namespacing is the enabling structure: optimization passes work on
+ these primitive nodes, not on Python expressions, which is what allows constant folding, dead-code
+ elimination, and kernel fusion to operate on TorchScript-compiled code.
+
```

---

### F-07 · `lst-state-dict-interface` — Listing 🟡

**Verbatim cite (L2731):**
> "Frameworks provide the `state_dict()` interface to access optimizer state for serialization (@lst-state-dict-interface), and resuming training requires loading both model parameters and optimizer state (@lst-checkpoint-save-load)."

**What is missing:** Both listings are cited as parenthetical asides in a single sentence. Neither receives a mechanism framing ("what to notice"). For `lst-state-dict-interface`: the key design choice is that `optimizer.state_dict()` exposes per-parameter state keyed by parameter identity, not name — which is why checkpoint portability across model changes requires careful key-mapping. For `lst-checkpoint-save-load`: the key design choice is that saving only `model.state_dict()` silently omits optimizer state, causing training to resume with reset momentum buffers even though weights are correct. Neither insight is in the surrounding prose.

**Missing move:** Add a mechanism sentence before each listing's parenthetical reference, or expand the single cite sentence to name what each listing demonstrates.

**Rule-compliant diff rewrite:**

```diff
- @Sec-model-training covers optimizer memory requirements and optimization strategies for large-scale
- training, where checkpoint size becomes a binding constraint. Frameworks provide the `state_dict()`
- interface to access optimizer state for serialization (@lst-state-dict-interface), and resuming
- training requires loading both model parameters and optimizer state (@lst-checkpoint-save-load).
+ @Sec-model-training covers optimizer memory requirements and optimization strategies for large-scale
+ training, where checkpoint size becomes a binding constraint. The `state_dict()` interface exposes
+ optimizer internals as a serializable dictionary keyed by parameter index
+ (@lst-state-dict-interface); crucially, this state is separate from model weights, so a checkpoint
+ that saves only `model.state_dict()` will restore correct weights but silently reset momentum
+ buffers, causing the optimizer to restart from a cold state. Restoring a complete training run
+ therefore requires saving and loading both (@lst-checkpoint-save-load).
```

---

### F-08 · `lst-checkpoint-save-load` — Listing 🟡

**Verbatim cite:** Same as F-07 (both cited in the same sentence at L2731).

**What is missing:** See F-07. `lst-checkpoint-save-load` specifically shows the pattern of packaging `model.state_dict()` and `optimizer.state_dict()` together in one dictionary before calling `torch.save()`. The key design choice — why both must be saved together and restored together — is not in the lead-in or lead-out. The payoff at L2774 discusses the history of AD systems and does not reference the checkpoint listing.

**Missing move:** See F-07 rewrite above, which covers both listings.

---

### F-09 · `lst-nested_modules` — Listing 🟡

**Verbatim cite (L4130, backward pointer — float defined at L4055):**
> "@Lst-nested_modules demonstrates how the module tree enables both recursive parameter access and hierarchical state serialization."

**What is missing:** The listing appears at L4055 (before its citation), so the float precedes its introduction — violating the forward-reference contract. The backward pointer at L4130 names two capabilities (recursive parameter access, hierarchical serialization) but does not name the mechanism or design choice in the code. The code shows an `nn.ModuleList` embedding `ResidualBlock` subtrees inside `ResNet`, with `model.state_dict()` flattening the tree to dotted-path keys. The key observation — that `nn.ModuleList` registration is what causes dotted-path keys like `blocks.0.conv1.weight` to appear rather than flat `conv1_weight` — is never stated in prose.

**Missing move:** Restructure so the cite comes before the float, and name the mechanism (ModuleList registration + dotted-path flattening as the enabling design choice).

**Rule-compliant diff rewrite** (replace the paragraph at L4130, moving the float to after this paragraph):

```diff
- The state_dict() method produces a flat key-value mapping of the full module tree, where dotted path
- names (for example, blocks.0.conv1.weight) encode the hierarchy. [...] @Lst-nested_modules
- demonstrates how the module tree enables both recursive parameter access and hierarchical state
- serialization.
+ The state_dict() method produces a flat key-value mapping of the full module tree, where dotted path
+ names (for example, blocks.0.conv1.weight) encode the hierarchy. The mechanism that produces those
+ dotted paths is `nn.ModuleList` registration: when a submodule is assigned as an attribute or
+ registered in a list, the framework records its position in the tree, and state_dict() walks that
+ tree depth-first to construct the path prefix. @Lst-nested_modules makes this visible: the ResNet
+ wraps several ResidualBlock instances inside an nn.ModuleList, and the printed state_dict keys
+ carry the resulting `blocks.0.`, `blocks.1.` path prefix for each block's weights.
```

---

### F-10 · `lst-parameter_freezing` — Listing 🟡

**Verbatim cite (L4134, parenthetical):**
> "Two practical patterns show how the principles become system controls: *selective parameter freezing* reduces unnecessary gradient work for transfer learning (@lst-parameter_freezing), and *module hooks* provide noninvasive inspection (@lst-module_hooks)."

**What is missing:** The parenthetical asides name the purpose of each listing but do not frame the mechanism or name what to notice in the code. For `lst-parameter_freezing`: the key design choice is that setting `requires_grad=False` on a submodule propagates recursively through all its parameters (via `.parameters()`) and excludes them from the autograd tape, so the backward pass never enters the frozen subgraph — not just that it saves "gradient work" in the abstract. The code's specific pattern (freeze `model.features`, unfreeze `model.classifier`) and the implication (autograd never constructs graph nodes for frozen layers) is not mentioned.

**Missing move:** Expand the lead-in to name the mechanism and what to look at in the listing.

**Rule-compliant diff rewrite:**

```diff
- Two practical patterns show how the principles become system controls: *selective parameter freezing*
- reduces unnecessary gradient work for transfer learning (@lst-parameter_freezing), and *module hooks*
- provide noninvasive inspection (@lst-module_hooks).
+ Two practical patterns show how the principles become system controls. The first is selective parameter
+ freezing: setting `requires_grad=False` on a submodule propagates through all its descendant
+ parameters and removes them from the autograd tape, so the backward pass never constructs graph nodes
+ for the frozen subgraph. @Lst-parameter_freezing shows the pattern for transfer learning, where the
+ pretrained feature extractor is frozen and only the classifier head receives gradients. The second
+ pattern is module hooks, which intercept intermediate computations without modifying model code
+ (@lst-module_hooks).
```

---

### F-11 · `lst-module_hooks` — Listing 🟡

**Verbatim cite (L4134, parenthetical) and secondary cite (L4159):**
> "Module hooks are the inspection counterpart to parameter freezing: they intercept intermediate computations without modifying model code, enabling gradient flow diagnosis and activation monitoring. @Lst-module_hooks illustrates both hook types."

**What is missing:** The secondary cite (L4159) is better than the parenthetical (which just says "noninvasive inspection"). But even L4159 names what hooks do ("intercept intermediate computations") without naming what the reader should look at in the code: the two hook types are `register_forward_hook` (receives input/output tensors) and `register_backward_hook` / `register_full_backward_hook` (receives gradient tensors). The design choice — that hooks run as callbacks at specific traversal points without modifying model source — is mentioned abstractly but the reader is not told which lines in the listing show each hook type or what distinguishes them.

**Missing move:** Add one sentence before or after the cite naming the two hook types and what each captures.

**Rule-compliant diff rewrite:**

```diff
- Module hooks are the inspection counterpart to parameter freezing: they intercept intermediate
- computations without modifying model code, enabling gradient flow diagnosis and activation monitoring.
- @Lst-module_hooks illustrates both hook types.
+ Module hooks are the inspection counterpart to parameter freezing: they intercept intermediate
+ computations without modifying model code, enabling gradient flow diagnosis and activation monitoring.
+ @Lst-module_hooks illustrates both hook types: a forward hook registered with
+ `register_forward_hook` receives the layer's input and output tensors after each forward pass,
+ making it suitable for activation monitoring; a backward hook registered with
+ `register_full_backward_hook` receives the gradient tensors flowing through the layer, making it
+ suitable for diagnosing vanishing or exploding gradients.
```

---

### F-12 · `lst-pipeline-parallelism-streams` — Listing 🟡

**Verbatim cite (L3416, backward pointer — float defined at L3392):**
> "Different model stages on separate GPUs can process different microbatches concurrently, with each stage's computation overlapping the next stage's data reception (see @lst-pipeline-parallelism-streams)."

**What is missing:** The float appears before its citation (L3392 vs. L3416), violating the forward-reference contract. The cite is a parenthetical backward pointer that names the capability (concurrent stage processing, overlapping computation and data reception) but does not name the mechanism in the code. The key design choice in the listing is the use of `events[stage_idx - 1][mb].wait()` as the inter-stage synchronization: each stage waits only for the previous stage's event on the same microbatch, not for the whole device — which is precisely what enables the overlap. This mechanism is not stated in the surrounding prose.

**Missing move:** Restructure so the cite precedes the float, and add a sentence naming the `event.wait()` synchronization as the enabling pattern.

**Rule-compliant diff rewrite** (rearrange: move the listing after the prose, and expand the introduction):

```diff
- This overlap principle extends naturally to model-stage overlap within a single node. Different model
- stages on separate GPUs can process different microbatches concurrently, with each stage's computation
- overlapping the next stage's data reception (see @lst-pipeline-parallelism-streams).
+ This overlap principle extends naturally to model-stage overlap within a single node. Different model
+ stages on separate GPUs can process different microbatches concurrently: each stage's computation
+ overlaps the next stage's data reception because stages synchronize only on a per-microbatch event
+ rather than waiting for the full device. @Lst-pipeline-parallelism-streams shows the pattern: each
+ stage issues `events[stage_idx - 1][mb].wait()` before processing microbatch `mb`, blocking only
+ on the upstream stage for that specific microbatch rather than serializing the whole pipeline.
```

---

### F-13 · `lst-dataloader-throughput` — Listing 🟡

**Verbatim cite (L3535):**
> "The DataLoader configuration is useful only when each parameter is tied to a bottleneck. @Lst-dataloader-throughput shows a typical setup where `num_workers` enables parallel loading, `prefetch_factor` controls pipeline depth, and `pin_memory` enables DMA transfers."

**What is missing:** The cite names the three parameters but frames the listing as a "typical setup" rather than naming what mechanism or design choice to look at. The prose after the float (L3537) continues with trade-off discussion but does not extract a "what to notice" moment from the code — specifically, that the three parameters address three different system-level bottlenecks (CPU I/O parallelism, prefetch queue depth, host-to-device DMA path), and that misconfiguring one while tuning another achieves nothing (e.g., high `num_workers` cannot help if `pin_memory=False` still forces intermediate copies). The "useful only when each parameter is tied to a bottleneck" framing in the cite sentence is the right instinct but is not developed into a mechanism statement.

**Missing move:** Strengthen the cite to name the mechanism (three distinct bottleneck axes) and add a payoff observation.

**Rule-compliant diff rewrite:**

```diff
- The DataLoader configuration is useful only when each parameter is tied to a bottleneck.
- @Lst-dataloader-throughput shows a typical setup where `num_workers` enables parallel loading,
- `prefetch_factor` controls pipeline depth, and `pin_memory` enables DMA transfers.
+ Each DataLoader parameter targets a different system bottleneck, and tuning one while ignoring the
+ others achieves only partial relief. @Lst-dataloader-throughput shows the three-parameter pattern:
+ `num_workers` multiplies CPU cores to overlap I/O and preprocessing, `prefetch_factor` deepens
+ the prefetch queue to hide load latency behind computation, and `pin_memory` moves host memory
+ to page-locked pages so the GPU's copy engine can initiate DMA transfers without the runtime
+ staging through a temporary buffer. The combination of all three is what sustains GPU utilization;
+ dropping any one reintroduces the bottleneck the others cannot compensate for.
```

---

### F-14 · `lst-overlap-compute-transfer` — Listing 🟡

**Verbatim cite (L3357):**
> "By placing data transfers on one stream and computation on another, the effective latency approaches the theoretical minimum of $\max(\text{compute\_time}, \text{transfer\_time})$ rather than their sum. Stream-based overlap effectively hides the $D_{\text{vol}}/\text{BW}$ penalty when computation is the longer operation (see @lst-overlap-compute-transfer):"

**What is missing:** The cite is parenthetical with "see" — a pointer, not a mechanism framing. The listing demonstrates the `non_blocking=True` + pinned memory pattern and the stream context manager. The key design choice in the code — that `non_blocking=True` returns immediately, and the transfer only becomes truly async when the *next* compute operation on the default stream has a dependency, unless explicit synchronization is used — is not mentioned. The payoff at L3390 explains `non_blocking=True` and pinned memory requirements, which is better than most, but the connection to the code lines is implicit.

**Missing move:** Convert the parenthetical to an active forward-reference, and name the key call in the listing.

**Rule-compliant diff rewrite:**

```diff
- Stream-based overlap effectively hides the $D_{\text{vol}}/\text{BW}$ penalty when computation is
- the longer operation (see @lst-overlap-compute-transfer):
+ Stream-based overlap effectively hides the $D_{\text{vol}}/\text{BW}$ penalty when computation is
+ the longer operation. @Lst-overlap-compute-transfer implements the pattern: a dedicated transfer
+ stream issues `tensor.to(device, non_blocking=True)` while the compute stream proceeds in parallel,
+ with the key call being `stream.wait_stream()` to enforce happens-before before the computation
+ that consumes the transferred data. Notice that `non_blocking=True` has no effect without pinned
+ host memory, since the copy engine requires page-locked pages for DMA.
```

---

### F-15 · `lst-cuda-events` — Listing 🟡

**Verbatim cite (L3422):**
> "CUDA events provide the alternative: fine-grained synchronization that blocks only the dependent stream, allowing other streams and the CPU to continue execution (see @lst-cuda-events):"

**What is missing:** The cite uses "see" — parenthetical pointer. The lead-in names the property (blocks only the dependent stream) but not the mechanism or the specific call to notice. The key design choice in the code is the two-step pattern: `event.record()` in the producer stream captures a point in time, and `stream.wait_event(event)` in the consumer stream inserts a dependency without blocking the CPU or other streams. This two-step mechanism (record then wait) is what distinguishes fine-grained from full-device sync — and it is not named in the lead-in.

**Missing move:** Convert to a forward-reference and name the two-step record/wait mechanism.

**Rule-compliant diff rewrite:**

```diff
- CUDA events provide the alternative: fine-grained synchronization that blocks only the dependent
- stream, allowing other streams and the CPU to continue execution (see @lst-cuda-events):
+ CUDA events provide the alternative: fine-grained synchronization that blocks only the dependent
+ stream, allowing other streams and the CPU to continue. @Lst-cuda-events shows the two-step
+ mechanism: `event.record()` in the producer stream captures a completion point in the stream's
+ command queue, and `stream.wait_event(event)` in the consumer stream inserts a hardware-level
+ dependency that stalls only the consumer until the recorded point is reached. The CPU issues both
+ calls without blocking, so the CPU-side loop and all non-dependent streams continue unimpeded
+ while the GPU enforces the ordering.
```

---

## Floats graded ✅ (no action needed)

| Label | Note |
|:------|:-----|
| `alg-reverse-mode-ad-trace` | Full walkthrough at L2322, cost analysis at L2320, purpose and invariant at L2271. |
| `eq-execution-continuum` | Takeaway ("each step rightward sacrifices flexibility for performance") at L1770. |
| `eq-compile-breakeven` | Symbol names, numeric evaluation, and implication (pays off within first epoch) all present. |
| `eq-dispatch-overhead` | All symbols named, consequence (overhead-bound regime) stated at L2000, second use correct. |
| `fig-mlfm-timeline` | "Gain productivity but lose transparency" takeaway at L136; payoff at L190 distills the layering principle. |
| `fig-comp-graph` | Mechanism and "execution problem turns on when graph is constructed" stated at L294. |
| `fig-mlfm-comp-graph` | Both panels narrated; "graph exists independently of execution" insight stated at L326. |
| `fig-mlfm-dynamic-graph-flow` | "Alternating pattern: define, execute" and Python-debugger advantage named at L604. |
| `fig-mlfm-static-graph` | Both phases narrated; ahead-of-time optimization enabled by the separation stated at L917. |
| `fig-python-tax` | Gaps named and explained, "compilation fuses into single kernel launch" stated at L1532. |
| `fig-compilation-continuum` | Crossover points and slope meanings explicitly walked through at L1898. |
| `fig-tensor-data-structure-a` | Hierarchy generalization and "adds one axis of organization" stated; payoff carries layout implication. |
| `fig-tensor-data-structure-b` | Rank-3 image representation and layout implication for convolutional layers stated at L3122. |
| `fig-tensor-memory-layout` | "Follow the same six values as they map into two different linear orderings" instructions plus stride explanation. |
| `fig-3d-parallelism` | Three placement dimensions named in prose; hardware memory implication previewed at L3622. |
| `fig-mlfm-core-ops` | All three groups named and purpose stated; "The prose follows that build-up" orientation given. |
| `fig-tensorflow-architecture` | Both pipeline columns walked through; deployment path interpretation at L4287. |
| `fig-onnx` | Hub-and-spoke model named; "notice how ONNX sits at the center" observation stated at L4465. |
| `lst-autograd-tape-example` | Two-node autograd tape and what it records stated; payoff at L581 names the result. |
| `lst-tf-static-graph` | Two-phase separation explained; symbolic vs. computed values contrast at L941. |
| `lst-torchscript-trace` | Tracing as "record every tensor operation" stated; limitation foreshadowed at L1292. |
| `lst-tracing-silent-failure` | Silent failure mode named, production consequence (wrong outputs for months) stated at L1318. |
| `lst-torchscript-conditional` | "Both paths in the IR" mechanism stated at L1344; contrast with tracing explicit. |
| `lst-torch-compile-intro` | "Captures hot paths, compiles to optimized kernels" stated; compilation overhead at L1423. |
| `lst-graph-break-control-flow` | Break mechanism explained; marshalling cost named at L1651. |
| `lst-torch-compile-benchmark` | Three-part measurement discipline (CUDA sync, warmup, iterations) stated at L1671; payoff at L1712 names each. |
| `lst-auto_diff_intro` | Systems challenge framed at L2193; concrete numeric trace for the same function at L2322. |
| `lst-forward_mode_ad` | Mechanism ("propagates derivatives alongside every operation") and 2× cost stated at L2219. |
| `lst-forward_mode_dual` | Dual-number mechanism stated; payoff at L2259 names per-input overhead. |
| `lst-reverse_simple_nn` | Both forward-store and backward-consume phases stated at L2330; three implementation requirements at L2357. |
| `lst-reverse_memory` | "Scales linearly with depth" mechanism stated; payoff at L2419 introduces checkpointing. |
| `lst-checkpoint-recompute` | "Forward pass keeps only segment boundaries; backward pass re-runs" contract stated at L2419. |
| `lst-grad-fn-chain` | `grad_fn` attribute and `next_functions` chain mechanism stated at L2458. |
| `lst-custom-autograd-function` | "Developer explicitly specifies what to save" contract stated at L2578. |
| `lst-autocast-usage` | `autocast` + `GradScaler` interaction and FP16 gradient underflow problem stated at L2626. |
| `lst-bf16-training` | Mechanism ("GradScaler calls all disappear") and BF16 exponent-range reason stated at L2657. |
| `lst-parameter_registration` | "Attribute assignment triggers registration" mechanism stated; "avoiding per-parameter Python bookkeeping" efficiency win stated at L3994. |
| `lst-jax-transformations` | Functional paradigm, "program transformations that can compose" stated at L4305. |
| `lst-framework-hello-world` | "Same simple network exposes how each design philosophy shapes the code" stated; lead-out at L4402 names the three-problem differences. |
| `lst-training-step-anatomy` | All five framework-stack activities named; "Tracing each phase reveals the three problems in action" stated at L4566. |
| `tbl-compile-modes` | "Trade between compilation time and runtime aggressiveness" stated; backend trade-off payoff at L1667 names TorchInductor, ONNX Runtime, TensorRT. |
| `tbl-framework-execution-models` | "Hybrid JIT achieves most of static graph performance while preserving much of eager execution's flexibility" stated at L1728. |
| `tbl-framework-archetype-strategy` | "Optimal strategy depends on which iron law term dominates" stated; payoff at L1886 applies to three concrete regimes. |
| `tbl-autograd-control` | "Callback-based extensions that autograd engine invokes at specific traversal points" stated; each mechanism mapped to cost at L2566. |
| `tbl-device-transfer-overhead` | Orders-of-magnitude variation stated; "transfer overhead can exceed computation time entirely" stated at L3336 and L3349. |
| `tbl-frameworks-parameter-discovery-apis` | "Durable pattern: trainable-parameter accessor + separate nontrainable-state channel" stated at L4041. |
| `tbl-mlfm-comparison` | "What each design lets the system see and optimize" framing; three-problem mapping stated at L4343. |
| `tbl-framework-efficiency-matrix` | "Constraint map: each row shows what the runtime gives up" stated; 2–10× latency range and failure modes at L4444. |
| `tbl-deployment-frameworks` | "Hard constraint rather than a late packaging step" stated at L4463; microcontroller tier exemplified. |
| `tbl-tf-comparison` | "Progressive constraint leading to progressive optimization" takeaway at L4504; operator-count drop quantified. |
| `tbl-tf-sw-comparison` | "TF Lite Micro eliminates the OS requirement entirely" and memory-mapped access stated at L4532. |
| `tbl-tf-hw-comparison` | "Binary size spans three orders of magnitude" and three architecture tiers stated at L4546. |
| `tbl-training-step-roofline` | "Overhead-bound" conclusion stated at L4869 with A100 compute/memory/overhead breakdown. |
