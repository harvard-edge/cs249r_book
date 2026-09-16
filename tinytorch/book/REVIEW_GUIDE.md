# Reviewer & Practitioner Guide: TinyTorch

**TinyTorch: The xv6 of Machine Learning Systems**

Prof. Vijay Janapa Reddi — Harvard University

## Purpose of the Commentary

A reader should be able to follow a computation from its entry point through the TinyTorch functions that implement it, explain the state those functions read or change, and predict the result for a small input. The xv6 analogy describes this relationship between a working reference and its commentary. It does not imply equivalent implementation languages, hardware control, or production coverage.

TinyTorch implements framework mechanisms in Python using NumPy for array storage and numerical operations. The current `src/` files contain complete instructor solutions, inline tests, and demonstrations. The narrative book explains those solutions; generated notebooks support execution and experiments. Reviewing solution stripping or assignment-completion enforcement is separate from reviewing the correctness and teachability of this reference.

## Reading Flows and Evidence

The book follows the source module order, with milestones that test composition. Each flow should begin with a concrete computation and finish with a contract the next flow can use.

| Flow | Modules | Execution path to follow | Evidence of understanding |
| :--- | :--- | :--- | :--- |
| Forward computation | @sec-tensors through @sec-dataloader | Tensor operation → activation → layer → loss, with batches supplied by a dataset and loader | Trace values and shapes; distinguish parameters from activations and array storage from Tensor wrappers. |
| Learning | @sec-autograd through @sec-training, @sec-milestone-1 | Recorded operation → backward traversal → parameter gradients → optimizer update | Explain broadcast reduction, graph lifetime, gradient accumulation, and normalization by the actual sample count. |
| Vision | @sec-convolutions | Image batch → convolution → pooling and normalization → loss | Trace one window and the contributions to its gradients; explain shared parameters. |
| Language | @sec-tokenization through @sec-transformers, @sec-milestone-2 | Text → token IDs → embeddings → attention → transformer outputs | Keep token, batch, head, and feature axes distinct; trace a causal attention computation before generation. |
| Measurement and optimization | @sec-profiling through @sec-acceleration | Baseline measurement → one transformation → correctness comparison → measurement | Separate elapsed time from analytical counts and distinguish an algorithmic demonstration from a changed storage or hardware execution format. |
| Reuse and evaluation | @sec-memoization through @sec-extensions, @sec-milestone-3 | Cache update → cached attention → controlled comparison → integrated evaluation | Identify cached state, reset and capacity behavior; explain remaining context-dependent work and verify the metric used to claim an improvement. |

The local trace and the integrated workload answer different questions. The trace explains the mechanism; the workload tests whether it honors the interfaces on both sides. Neither a passing assertion nor a plausible explanation alone is sufficient evidence for both.

## Implementation Boundaries to Preserve

The production bridge can discuss alternatives beyond TinyTorch, provided it identifies them as alternatives. Review claims against these boundaries before checking their style:

- TinyTorch's Tensor constructor creates float32 NumPy array storage. A NumPy view inside an operation does not imply that the returned Tensor shares that storage. Zero-copy view claims require evidence from the complete wrapping path.
- The reference DataLoader forms batches synchronously. Worker processes, asynchronous prefetch queues, pinned memory, and device transfers belong to production comparisons.
- Reference convolution exposes sliding-window loops. `im2col` and hardware-specific convolution kernels are alternatives, not the implementation of its forward path.
- Quantization demonstrates integer codes, scale and zero point, calibration, and reconstruction. Tensor storage remains float32; modeled INT8 storage savings are not measurements of a packed integer Tensor or an integer matrix-multiplication kernel.
- Compression includes pruning, low-rank factorization, and knowledge distillation. Low-rank factorization is not an implemented LoRA training system. A student distillation loss must remain differentiable while teacher outputs remain fixed.
- NumPy fusion and tiling experiments do not establish register residency, SRAM placement, GPU synchronization, or a universal speedup. Timing claims need the measured workload and comparison conditions.
- The KV cache stores earlier keys and values in preallocated capacity with a tracked position. It is not a ring-buffer implementation. Reusing projections does not remove attention over the growing prefix or guarantee constant decoding latency.
- Benchmarking and the capstone provide educational comparisons. They do not certify MLPerf compliance, and separately measured speedups do not establish the performance of their composition.

@sec-extensions extends the discussion to production tools and hardware. Its examples must remain visibly outside the NumPy reference implementation and must not become undeclared prerequisites for earlier chapters.

## Review Procedure

Choose one execution path before reading a chapter in detail. The following sequence tests whether the explanation can guide a reader through the code:

1. **Verify the Hourglass Arc**: Confirm the chapter opens with the foundational machine learning motivation (the mathematical necessity or physical bottleneck) before presenting code, narrows into the inspectable implementation and worked numerical trace, and expands at the conclusion into a production systems bridge.
2. **Follow Shapes, State, and Invariants**: Identify the caller and source symbol that receives control. Record input and output shapes, persistent state, and ephemeral call state. Follow actual dispatch rather than inferring behavior from a familiar API name.
3. **Trace Intermediate Numbers by Hand**: State the invariant and work through the chapter's small trace independently. Check intermediate values, reduction axes, mutation, and graph connectivity where relevant.
4. **Check Progressive Disclosure and Marginal Signposts**: Verify that prerequisites are declared or linked via marginal back-links (`↩ Jump back: @sec-...`), and that forward-looking optimizations are flagged with forward-links (`↪ Jump ahead: @sec-...`). Confirm that marginal micro-plots and semantic callout tiers (Napkin Math `nbk-`, Systems Perspective `psp-`, Checkpoint `chk-`, War Story `wr-`) are positioned accurately.
5. **Probe Boundary Cases**: Find a test that would fail if the invariant were violated. Check a meaningful boundary case, such as a shared graph, an unequal final batch, a constant quantization range, or cache exhaustion.
6. **Evaluate the Production Bridge**: Read the production bridge after the reference path. Verify that it explains the constraint driving a different design and distinguishes measured results, estimates, and hypothetical examples.

Record findings with the chapter and source location, the disputed claim, a concrete consequence for the reader, and a proposed correction. Numerical or execution claims should include a small reproducer or a supporting test. Editorial suggestions should name the comprehension problem they solve.

## Validation and Source Authority

Registered book listings are extracted from source symbols through `tools/listings.py` and `tools/listings.yml` in the narrative-book directory. Run `python3 tools/listings.py --check` there to detect drift. A successful listing check establishes agreement for registered excerpts; surrounding explanations and worked traces still require review.

From the TinyTorch package root, in its development environment, `python3 tools/check_reference.py` checks targeted regressions against a temporary source-built package. After regenerating the instructor exports with `python3 -m tito.main dev export --all`, `python3 tools/release_check.py` checks structure, tests, and notebook execution in progressive isolation. Regenerating exports replaces generated notebook content, so use a reference checkout for this workflow.

Report the checks run and their results alongside any remaining uncertainty. A complete instructor reference provides a stable object to study, but a green suite does not justify claims about untested inputs, unimplemented hardware behavior, or learning outcomes that have not been evaluated with students.
