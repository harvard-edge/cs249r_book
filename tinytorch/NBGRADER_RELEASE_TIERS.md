# TinyTorch NBGrader Release Tiers

Internal proposal for using Tito and NBGrader to create instructor,
student, and challenge artifacts without weakening the TinyTorch
narrative flow.

This is a design document, not a public user guide.

## Pedagogical Goal

The student release should not mean "remove every solution." It should
mean "leave the learner with the smallest meaningful implementation set
that produces ownership of the system."

TinyTorch should use backward design:

1. Define the evidence of understanding for each module.
2. Strip only the implementation regions that produce that evidence.
3. Keep scaffolding, hints, setup, examples, visible tests, and repeated
   patterns when they reduce incidental load without removing the
   learning target.

This matches the existing Build, Use, Reflect flow. Students should feel:

- "I built a working ML systems stack."
- "I implemented the pieces that matter."
- "The surrounding structure helped me see the pattern."

They should not feel:

- "I copied boilerplate for every repeated method."
- "The assignment removed so much context that I cannot see the system."
- "I am graded on plumbing instead of the concept."

## Release Tiers

### Instructor

Complete reference state.

- Full solutions remain.
- Instructor notes, reference answers, and grading metadata remain.
- Used by course staff, not distributed as a student artifact.

### Student

Canonical public assignment.

- Core pedagogical implementation regions are stripped.
- Scaffolding remains.
- Hints and comments remain.
- Visible tests remain.
- Repeated patterns can remain solved after the first representative
  example.

The student tier is the primary teaching product.

### Challenge

Optimization extension tier.

This is not "less scaffolded student." Challenge means optimization work
on top of a working system: speed, memory, numerical stability,
compression, latency, throughput, and benchmarking.

For challenge artifacts, the baseline system should generally be working.
The stripped regions should be optimization tasks, not the same
fundamental implementation holes from the student tier.

## NBGrader Compliance Model

NBGrader should remain the grading and release engine. Tito should own
TinyTorch-specific policy before NBGrader runs.

Current flow:

```bash
tito nbgrader generate --all
tito nbgrader release --all
```

Proposed hidden/internal flow:

```bash
tito nbgrader generate --all --tier student
tito nbgrader generate --all --tier challenge
```

The flag should stay hidden from public help until the pedagogy is
settled.

NBGrader constraints:

- `metadata.nbgrader` must remain schema-compliant.
- Solution regions must stay in cells with a stable `grade_id` when
  NBGrader is expected to strip them.
- Written reflection answers can be stripped by marker regions while
  remaining ungraded, using the config Tito already writes:
  `c.ClearSolutions.enforce_metadata = False`.
- Custom TinyTorch policy should not be stored inside
  `metadata.nbgrader`, because NBGrader rejects unknown fields.

Recommended TinyTorch metadata:

```json
{
  "tinytorch": {
    "release_role": "core"
  }
}
```

Recommended region marker syntax:

```python
### BEGIN SOLUTION role=core
...
### END SOLUTION
```

NBGrader still recognizes this because the delimiter contains
`BEGIN SOLUTION` and `END SOLUTION`. Tito can parse the role first:

- `role=core`: leave markers for student release; NBGrader strips it.
- `role=scaffold`: remove marker lines, keep content for student release.
- `role=challenge`: keep for student baseline, strip for challenge.
- `role=instructor`: keep only in instructor artifacts.

Unannotated existing solution regions should default to `core` until a
module has been reviewed. That preserves current behavior.

## Student Release Standard

A student-facing notebook is ready when:

- The missing regions map to the module learning outcome.
- Each missing region has enough local context to implement it.
- Hints and comments remain outside stripped regions.
- Setup/import/export cells are preserved.
- Visible checks run without hidden instructor code.
- Repetition is intentional: if a pattern has already been learned, keep
  later instances as scaffold unless the variation is the concept.
- The final module and milestone can run once the core regions are filled.

## Module Classification Proposal

These classifications are proposed at the `grade_id` level. Some cells
contain multiple solution regions; those should be split by region role
when the marker pass happens.

### Foundation Tier

| Module | Evidence of understanding | Student core strips | Keep as scaffold | Challenge direction |
|---|---|---|---|---|
| 01 Tensor | Tensors hold data, transform shapes, and implement array math. | `tensor-class`: `__init__`, arithmetic, matmul validation, matmul, slicing, reshape, transpose, reductions | imports, representation helpers, memory helpers, simple properties | memory layout, broadcasting edge cases, operation fusion |
| 02 Activations | Nonlinearities transform tensors and require numerical care. | `sigmoid-impl`, `relu-impl`, `softmax-impl` | setup, `tanh-impl`, `gelu-impl` after one or two patterns are learned | stable approximations, vectorized kernels, fused activation paths |
| 03 Layers | Layers own parameters and define reusable forward transforms. | `linear-layer`: init, forward, parameters; `dropout-layer`: mask and forward behavior | imports, `layer-base`, repr methods, analysis helpers | initialization experiments, dropout variants, memory/perf analysis |
| 04 Losses | Losses define the training objective and must be numerically stable. | `log-softmax`, `mse-loss`, `cross-entropy-loss` | setup, `binary-cross-entropy-loss` as a related pattern, analysis helpers | stability stress tests, reduction modes, mixed precision behavior |
| 05 DataLoader | Data pipelines batch, shuffle, and collate samples into training-ready tensors. | `tensordataset-implementation`, `dataloader-implementation` | abstract `dataset-implementation`, most augmentation helpers, integration demo | augmentation policy, streaming, prefetching, memory pressure |
| 06 Autograd | Reverse-mode AD propagates gradients through a computation graph. | `_reduce_broadcast_grad`, `add-backward`, `mul-backward`, `matmul-backward`, `sum-backward`, one nonlinear backward path | repeated backward operators after the pattern is learned, `function-base`, `enable-autograd`, no-grad context, helper glue | graph memory, checkpointing, more complete operator coverage |
| 07 Optimizers | Optimizers turn gradients into parameter updates with state. | `optimizer-base`, `extract-gradient`, `sgd-optimizer`, `adam-update-moments`, `adam-step` | AdamW variants after Adam is learned, analysis helpers | optimizer comparisons, scheduler interactions, convergence diagnostics |
| 08 Training | Training loops orchestrate batches, loss, backward pass, optimizer updates, and validation. | `scheduler`, `gradient_clipping`, `trainer-process-batch`, `trainer-optimizer-update`, `trainer-train-epoch`, `trainer-evaluate` | `trainer-init`, checkpoint save/load, complete integration example | gradient accumulation strategies, fault tolerance, throughput tuning |

### Architecture Tier

| Module | Evidence of understanding | Student core strips | Keep as scaffold | Challenge direction |
|---|---|---|---|---|
| 09 Convolutions | Spatial layers compute feature maps with kernels, padding, pooling, and channel layout. | `conv2d-class`: output shape, padding, convolution loops, forward; `maxpool2d-class`: output shape, pool loops, forward | AvgPool after MaxPool, BatchNorm as scaffold unless explicitly taught, spatial analysis, `simple-cnn` glue | im2col, vectorized convolution, cache-friendly layout |
| 10 Tokenization | Text becomes stable integer sequences through vocabulary and merge rules. | `char-tokenizer`, `bpe-count-pairs`, `bpe-merge-pair`, core `bpe-tokenizer` training/apply/encode | base tokenizer interface, decode variants, tokenization utility wrappers | vocabulary trade-offs, compression ratios, tokenizer benchmarks |
| 11 Embeddings | Discrete IDs become learned vectors plus positional structure. | `embedding-backward`, `embedding-init` forward path, `posenc-sinusoidal-table`, `emblayer-forward` | repr/parameter boilerplate, helper constructor, repeated positional wrapper logic | sparse update efficiency, memory footprint, tied embeddings |
| 12 Attention | Attention computes scores, masks them, normalizes them, and composes values across heads. | `attn-compute-scores`, `attn-scale-scores`, `attn-apply-mask`, `attn-scaled-dot-product`, `multihead-attention` split/merge/forward | imports and parameters boilerplate | attention memory, causal masking variants, fused QKV paths |
| 13 Transformers | Transformers compose normalization, attention, MLPs, residuals, and generation. | `layer-norm` forward, `mlp` forward, `transformer-block` forward, `gpt` forward/generate path | constructor boilerplate, analysis helpers, integration demo | KV-cache integration, generation speed, parameter scaling experiments |

### Optimization Tier

The optimization modules are still part of the student learning path.
Challenge work should extend them, not replace their fundamentals.

| Module | Evidence of understanding | Student core strips | Keep as scaffold | Challenge direction |
|---|---|---|---|---|
| 14 Profiling | Performance work starts with measurement, not guessing. | `profiler_class`: parameter counting, FLOP counting, memory measurement, latency measurement, derived bottleneck metrics | quick profile helper, weight distribution helper, advanced report text | profiler overhead reduction, more accurate memory models, hardware counters |
| 15 Quantization | Precision reduction trades accuracy, memory, and speed through scale/zero-point calibration. | `quantize_int8`, `dequantize_int8`, `quantized_linear` calibrate/forward/memory, `measure_layer_bytes`, `analyze_model_sizes` | model traversal helpers, export wrapper, reflection prompts with stripped answers | per-channel quantization, dynamic quantization, accuracy-speed frontier |
| 16 Compression | Model size can be reduced through sparsity, rank, and distillation. | `measure-sparsity`, `magnitude-prune`, `low-rank-approx`, `distillation` loss | `structured-prune`, `compress-model-comprehensive`, profiler demo | structured sparsity kernels, pruning schedules, accuracy recovery |
| 17 Acceleration | Runtime optimization improves arithmetic intensity and reduces memory traffic. | `vectorized-matmul`, `fused-gelu`, `tiled-matmul` | `unfused-gelu` baseline, analysis helpers, profiler demo | tile tuning, cache behavior, fusion benchmarks |
| 18 Memoization | KV-cache trades memory for faster autoregressive generation. | `kvcache-class` update/get/memory, `kv-create-cache`, `kv-cached-attention`, `kv-cached-generate` | wrapper integration in `kv-enable-cache`, analysis prompts | eviction policies, batching, long-context memory management |
| 19 Benchmarking | Optimization claims need reproducible latency, accuracy, memory, and comparison metrics. | `benchmark-dataclass`, `timer-context`, `benchmark-init`, latency/accuracy/memory runs, `benchmark-compare` | plotting, report formatting, MLPerf-style compliance boilerplate | statistical rigor, leaderboard scoring, energy estimation |
| 20 Capstone | Students package a working system and measure improvements. | `toy-model`, `benchmark-report` latency/memory, submission generation path | example workflow, schema glue | optimization workflow, leaderboard track, final competition tasks |

## Implementation Plan

### Pass 1: Add Internal Role Support

- Add a hidden `--tier` option to `tito nbgrader generate`.
- Default to current behavior.
- Parse optional role annotations from solution marker lines.
- Keep role data in TinyTorch-owned metadata or marker syntax, not in
  `metadata.nbgrader`.
- Add unit tests for `core`, `scaffold`, `challenge`, and `instructor`
  handling.

### Pass 2: Pilot Three Modules

Pilot modules:

- Module 01 Tensor: large multi-region cell with clear core/scaffold
  split.
- Module 06 Autograd: highest repetition risk in the Foundation tier.
- Module 15 Quantization: includes code solutions and markdown
  reflection answers.

The pilot should prove:

- Student release strips only intended core regions.
- Scaffold regions keep code and comments.
- Markdown answers strip correctly.
- NBGrader `generate_assignment` succeeds.
- Visible tests remain runnable.

### Pass 3: Classify All Modules

Apply the role markers across all source modules using the table above.

Every module should have:

- A short learning-outcome statement.
- A list of student-core regions.
- A list of scaffold regions intentionally kept.
- Optional challenge regions.
- A validator report showing no malformed role markers.

### Pass 4: Add Release Validation

Extend the NBGrader validator with tier checks:

- No `role=core` region is left unstripped in student release.
- No `role=scaffold` region is stripped in student release.
- Challenge release has a working baseline plus stripped challenge
  targets.
- Instructor source remains complete.
- NBGrader schema remains valid.

Run these checks in CI or release verification:

```bash
python3 tests/validate_nbgrader_config.py
tito nbgrader generate --all --tier student
tito nbgrader release --all
```

When NBGrader is installed, the release smoke test should verify all 20
release notebooks contain no solution delimiters.

## Open Decisions

1. Should the hidden flag be `--tier`, `--profile`, or environment-only?
   Recommendation: hidden `--tier`, with public docs silent for now.
2. Should challenge artifacts include completed student-core solutions?
   Recommendation: yes, because challenge means optimization on a
   working baseline.
3. Should some reflection answers be manually graded?
   Recommendation: not until the instructor grading rubric is explicit.
   For now, strip instructor answers but do not assign points.
4. Should we preserve all current `grade_id`s?
   Recommendation: yes. Do not churn IDs unless a cell is split.

## References

- Backward design: https://teaching.uic.edu/cate-teaching-guides/syllabus-course-design/backward-design/
- NBGrader metadata: https://nbgrader.readthedocs.io/en/latest/contributor_guide/metadata.html
- NBGrader assignment creation and solution stripping: https://nbgrader.readthedocs.io/en/latest/user_guide/highlights.html
- Parsons/scaffolded programming tasks: https://textbooks.cs.ksu.edu/tlcs/3-cs-teaching-approaches/04-parsons-problems/
