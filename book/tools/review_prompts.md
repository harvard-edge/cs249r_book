# Volume 1 Final Review Prompts for Claude 4.6 Max

## Strategy

The review is split into **two phases**:

1. **Phase 1 — Per-Chapter Deep Dives** (16 parallel agents, one per chapter)
2. **Phase 2 — Cross-Chapter Coherence** (4 parallel agents, one per Part)

Each prompt is self-contained and can be run independently as a Cursor agent task.

---

## Phase 1: Per-Chapter Deep Dives

Each chapter gets its own agent. The agent reviews the chapter in isolation for
internal correctness. All 16 can run in parallel (or in batches of 4).

### Shared Preamble (include at the top of every Phase 1 prompt)

```
You are a technical reviewer doing a final pre-press review of a chapter from
"Introduction to Machine Learning Systems" (Volume I) by Vijay Janapa Reddi,
being published by MIT Press. This is the FINAL review before print — errors
found now will be permanently published.

CONTEXT FILES TO READ FIRST (in this order):
1. /Users/VJ/GitHub/mlsysbook-vols/CLAUDE.md — project conventions
2. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/physx/constants.py — the single
   source of truth for all hardware specs and physical constants
3. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/physx/formulas.py — formula helpers
4. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/parts/summaries.yml
   — book structure and part descriptions

Then read the chapter file specified below IN FULL. For chapters over 3000 lines,
read in chunks of 500 lines to ensure nothing is missed.

REVIEW CHECKLIST — evaluate every item and report findings organized by category:

## A. Factual & Technical Accuracy
- Are all technical claims correct? (e.g., "GPUs have thousands of cores" — verify scale)
- Are hardware specifications (FLOPS, bandwidth, memory, TDP) consistent with
  constants.py? Cross-check every number that appears in prose against the
  Python setup block and constants.py.
- Are year/date claims correct? (e.g., "Transformers were introduced in 2017")
- Are attribution claims correct? (e.g., "ResNet was proposed by He et al.")
- Are benchmark numbers plausible? (e.g., "achieves 76% top-1 accuracy on ImageNet")

## B. Calculation Verification
- Read the Python ```{python} blocks at the top of the chapter carefully.
- Trace every `{python} variable_name` inline reference back to its definition.
  Verify the calculation is correct (units, arithmetic, order of magnitude).
- Check that formatted strings use appropriate precision (not too many/few decimals).
- Verify unit conversions are correct (e.g., bytes to GB, FLOPS to TFLOPS).
- Check for any Markdown() usage and verify the LaTeX it produces is well-formed.
- Flag any inline references that appear to use the WRONG variable (e.g., using
  a100_bw where h100_bw was intended based on surrounding context).

## C. Internal Consistency
- Do numbers in the prose match numbers in tables within the same chapter?
- Do figures referenced in the text exist? (Check for @fig-* references)
- Do section cross-references within the chapter resolve? (Check @sec-* references)
- Are terms used consistently? (e.g., don't switch between "inference" and
  "prediction" without reason)
- Is notation consistent with the notation guide in
  contents/vol1/frontmatter/notation.qmd?

## D. Narrative Flow & Structure
- Does the chapter opening ("Purpose" section) accurately preview what follows?
- Do the Learning Objectives in the callout-tip match what the chapter delivers?
- Does each section transition logically to the next?
- Are there any sections that feel abrupt, unfinished, or disconnected?
- Does the chapter conclusion/summary accurately reflect the content?
- Are there any redundant passages that say the same thing twice?

## E. Cross-Reference Integrity
- List ALL @sec-* references that point OUTSIDE this chapter (to other chapters).
  For each, note the target section label so Phase 2 can verify they resolve.
- List ALL @fig-*, @tbl-*, @eq-* references. Verify internal ones resolve.
- List ALL @thm-*, @def-*, @exm-* references if any.

## F. LaTeX & Formatting
- Are all equations well-formed? (matched $...$ or $$...$$, balanced braces)
- Are \index{} entries reasonable and consistent in style?
- Are callout blocks properly opened and closed (matching ::: pairs)?
- Are figure captions descriptive and accessible (fig-alt attributes present)?
- Are tables properly formatted with headers?

## G. Language & Style
- Flag any grammatical errors, awkward phrasing, or unclear sentences.
- Flag any overly casual language inappropriate for a MIT Press textbook.
- Flag any sentences that are excessively long (>40 words) and hard to parse.
- Note any jargon used without definition.

## OUTPUT FORMAT
For each category (A-G), provide:
- ✅ PASS — if no issues found in that category
- ⚠️ ISSUES — list each issue with:
  - Line number or section reference
  - The problematic text (quoted)
  - What's wrong
  - Suggested fix (if applicable)

End with a SUMMARY section:
- Total issues found (by severity: Critical / Warning / Minor)
- Overall assessment of chapter readiness
- List of all outbound cross-references for Phase 2 verification
```

---

### Chapter Prompts (Phase 1)

Each prompt below should be prefixed with the Shared Preamble above.

---

#### 1. Introduction

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/introduction/introduction.qmd

ADDITIONAL CONTEXT:
- This is Chapter 1 and the reader's first encounter with the book's core ideas.
- It introduces the D·A·M taxonomy (Data, Algorithm, Machine) which is the
  conceptual spine of the entire book.
- It should set up the "why ML systems engineering matters" motivation.

SPECIAL ATTENTION:
- Verify the D·A·M taxonomy is introduced clearly and consistently — this
  framework is referenced in every subsequent chapter.
- Check that any "preview of upcoming chapters" text accurately matches the
  actual chapter titles and ordering in the config.
- Verify any historical timeline claims (Moore's Law dates, key milestones).
- This chapter has calculations from physx/ch_introduction.py — read that file
  and verify all computed values.
```

#### 2. ML Systems

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/ml_systems/ml_systems.qmd

ADDITIONAL CONTEXT:
- This chapter covers what makes ML systems different from traditional software.
- It should build on the D·A·M taxonomy from Chapter 1.

SPECIAL ATTENTION:
- Verify claims about ML system characteristics vs traditional software are
  accurate and well-supported.
- Check that any system architecture diagrams are correctly described in prose.
- Verify any performance/efficiency comparisons have correct numbers.
- Read physx/_legacy_ch_ml_systems.py if referenced.
```

#### 3. Workflow

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/workflow/workflow.qmd

ADDITIONAL CONTEXT:
- Covers the ML development lifecycle / workflow.
- Should connect the abstract concepts from Chapters 1-2 to practical process.

SPECIAL ATTENTION:
- Verify workflow stage descriptions are complete and in logical order.
- Check that any tool/framework mentions are current (not deprecated).
- Ensure terminology for workflow stages is used consistently.
```

#### 4. Data Engineering

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/data_engineering/data_engineering.qmd

ADDITIONAL CONTEXT:
- Covers data pipelines, preprocessing, augmentation, storage.
- Has its own supplementary bibliography: data_engineering.bib

SPECIAL ATTENTION:
- Verify any data volume claims (e.g., "ImageNet has 1.2M images") are correct.
- Check that data format descriptions (TFRecord, Parquet, etc.) are accurate.
- Verify any throughput/bandwidth calculations for data loading.
- Check the supplementary .bib file references resolve.
```

#### 5. Deep Learning Primer

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/dl_primer/dl_primer.qmd

ADDITIONAL CONTEXT:
- Mathematical foundations: loss functions, backpropagation, optimization.
- This chapter bridges math theory to the engineering chapters that follow.

SPECIAL ATTENTION:
- Verify ALL mathematical equations are correct (gradients, chain rule,
  update rules, loss functions).
- Check that notation matches the notation guide.
- Verify any numerical examples (e.g., "if learning rate is 0.01 and gradient
  is 0.5, the update is...") compute correctly.
- Ensure backpropagation walkthrough is mathematically rigorous.
```

#### 6. DNN Architectures

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/dnn_architectures/dnn_architectures.qmd

ADDITIONAL CONTEXT:
- Covers CNNs, RNNs, Transformers, and other architectures.
- Should connect architecture choices to systems implications.

SPECIAL ATTENTION:
- Verify parameter count calculations (e.g., "a Conv2d layer with 64 3x3
  filters has 576 parameters" — is that right? Should it be 64*3*3*C_in?).
- Verify FLOPs calculations for different layer types.
- Check that architecture diagrams match textual descriptions.
- Verify attention mechanism math (Q, K, V dimensions, scaling factor).
- Check that the Transformer "Attention is All You Need" description is accurate.
```

#### 7. Frameworks

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/frameworks/frameworks.qmd

ADDITIONAL CONTEXT:
- Covers PyTorch, TensorFlow, JAX, ONNX, and the framework ecosystem.
- Should explain computational graphs, autograd, JIT compilation.

SPECIAL ATTENTION:
- Verify code examples are syntactically correct and use current APIs.
- Check that framework comparison claims are fair and accurate.
- Verify computational graph explanations match the code.
- Flag any deprecated API usage.
```

#### 8. Training

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/training/training.qmd

ADDITIONAL CONTEXT:
- Covers distributed training, parallelism strategies, mixed precision.
- One of the longest chapters (~3565 lines) — be especially thorough.

SPECIAL ATTENTION:
- Verify data/model/pipeline parallelism descriptions are technically correct.
- Check communication overhead calculations (all-reduce, ring all-reduce).
- Verify mixed-precision training description (FP16 forward, FP32 master weights).
- Check any scaling law claims and their citations.
- Verify GPU memory calculations (model params + gradients + optimizer state +
  activations).
- Verify batch size / learning rate scaling relationships.
```

#### 9. Data Selection

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/data_selection/data_selection.qmd

ADDITIONAL CONTEXT:
- Covers coreset selection, curriculum learning, active learning.
- Has calculations in physx/ch_data_selection.py.

SPECIAL ATTENTION:
- Read physx/ch_data_selection.py and verify all computed values appear
  correctly in the chapter prose.
- Verify any efficiency claims (e.g., "using 10% of data achieves 95% accuracy").
- Check that theoretical foundations (submodularity, etc.) are correctly stated.
```

#### 10. Model Compression

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/optimizations/model_compression.qmd

ADDITIONAL CONTEXT:
- Covers pruning, quantization, knowledge distillation, NAS.
- The LONGEST chapter (~5698 lines) — read in 500-line chunks carefully.

SPECIAL ATTENTION:
- Verify quantization math: dynamic range calculations, scale/zero-point
  formulas, the mapping from FP32 to INT8.
- Verify pruning calculations: sparsity ratios, structured vs unstructured
  speedup expectations.
- Check knowledge distillation loss function formulations.
- Verify NAS search space size calculations if present.
- Verify any compression ratio claims (e.g., "4x compression from FP32 to INT8").
- Cross-check all hardware specs with constants.py.
```

#### 11. Hardware Acceleration

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd

ADDITIONAL CONTEXT:
- Covers GPUs, TPUs, NPUs, systolic arrays, roofline model.
- Heavy use of physx calculations — has ch_hw_acceleration.py.
- Central to the "Machine" axis of D·A·M.

SPECIAL ATTENTION:
- Read physx/ch_hw_acceleration.py AND constants.py together.
- Verify EVERY hardware specification in the chapter against constants.py:
  FLOPS, bandwidth, memory capacity, TDP for V100, A100, H100, T4, TPUv4.
- Verify roofline model calculations: ridge point = peak_compute / peak_bandwidth.
- Verify arithmetic intensity calculations for example workloads.
- Check that dataflow strategy descriptions (weight/output/input stationary)
  are technically accurate.
- Verify energy cost hierarchy (register < SRAM < DRAM < off-chip).
- Verify interconnect bandwidth numbers (NVLink, InfiniBand).
```

#### 12. Benchmarking

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/benchmarking/benchmarking.qmd

ADDITIONAL CONTEXT:
- Covers MLPerf, benchmarking methodology, metrics.
- Should connect to hardware acceleration (previous chapter).

SPECIAL ATTENTION:
- Verify MLPerf benchmark descriptions match the actual MLPerf specifications.
- Check that metrics definitions are correct (throughput, latency, energy).
- Verify any benchmark result numbers cited.
- Check that benchmarking methodology advice is sound.
```

#### 13. Serving / Inference

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/serving/serving.qmd

ADDITIONAL CONTEXT:
- Covers model serving, inference optimization, deployment patterns.
- First chapter in Part IV: Deploy.

SPECIAL ATTENTION:
- Verify latency calculations (network + compute + pre/post processing).
- Check that serving architecture descriptions (batching, model loading) are
  accurate.
- Verify any SLA/latency budget calculations.
- Check that optimization techniques described actually work as claimed.
```

#### 14. ML Operations

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/ops/ops.qmd

ADDITIONAL CONTEXT:
- Covers MLOps, CI/CD for ML, monitoring, model versioning.
- Should build on the serving chapter.

SPECIAL ATTENTION:
- Verify that MLOps tool descriptions are accurate and current.
- Check that monitoring metric descriptions are correct.
- Verify any claims about model drift detection methods.
- Ensure the distinction between DevOps and MLOps is accurately drawn.
```

#### 15. Responsible Engineering

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd

ADDITIONAL CONTEXT:
- Covers fairness, bias, privacy, environmental impact, ethical considerations.
- Should integrate themes from all previous chapters.

SPECIAL ATTENTION:
- Verify fairness metric definitions are mathematically correct.
- Check that privacy technique descriptions (DP, federated learning) are accurate.
- Verify any carbon footprint / energy consumption calculations.
- Ensure claims about legal/regulatory frameworks are current.
- Check that bias examples are well-sourced and accurately described.
```

#### 16. Conclusion

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/conclusion/conclusion.qmd

ADDITIONAL CONTEXT:
- Wraps up Volume I and previews Volume II.
- Should synthesize themes from all chapters.

SPECIAL ATTENTION:
- Verify that the conclusion accurately summarizes what each chapter covered.
- Check that any forward-looking claims about Vol II are accurate/appropriate.
- Verify that the D·A·M framework is properly tied back together.
- Ensure no chapters are omitted from the synthesis.
```

---

### Appendix Prompts (Phase 1, continued)

#### A1. D·A·M Taxonomy Appendix

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/backmatter/appendix_dam.qmd

SPECIAL ATTENTION:
- This appendix formally defines the D·A·M framework used throughout the book.
- Verify internal consistency with how D·A·M is used in all chapters.
```

#### A2. Machine Appendix

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/backmatter/appendix_machine.qmd

ADDITIONAL CONTEXT: Read physx/constants.py first.

SPECIAL ATTENTION:
- Verify ALL hardware specifications match constants.py.
- Check calculation correctness for any derivations.
```

#### A3. Algorithm Appendix

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/backmatter/appendix_algorithm.qmd

ADDITIONAL CONTEXT: Read physx/ch_appendix_algorithm.py first.

SPECIAL ATTENTION:
- Verify all computed values from ch_appendix_algorithm.py.
- Check mathematical correctness of algorithm descriptions.
```

#### A4. Data Appendix

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/backmatter/appendix_data.qmd

SPECIAL ATTENTION:
- Verify dataset statistics are correct.
- Check any data-related calculations.
```

#### A5. Glossary

```
CHAPTER: /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/backmatter/glossary/glossary.qmd

SPECIAL ATTENTION:
- Verify definitions are technically accurate.
- Check for terms used in chapters but missing from the glossary.
- Check for glossary entries that seem unused.
- Verify alphabetical ordering.
```

---

## Phase 2: Cross-Chapter Coherence Reviews

Run these AFTER Phase 1. Each reviews one Part for cross-chapter flow.
Can run all 4 in parallel.

### Shared Preamble (include at the top of every Phase 2 prompt)

```
You are a senior technical editor doing a cross-chapter coherence review of
"Introduction to Machine Learning Systems" (Volume I) for MIT Press. Phase 1
individual chapter reviews have already been completed. Your job is to verify
that chapters within this Part flow together coherently, and that cross-references
to other Parts are valid.

CONTEXT FILES TO READ FIRST:
1. /Users/VJ/GitHub/mlsysbook-vols/CLAUDE.md
2. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/parts/summaries.yml
3. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/physx/constants.py
4. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/config/_quarto-pdf-vol1.yml
   (lines 62-119 for chapter ordering)

CROSS-CHAPTER REVIEW CHECKLIST:

## A. Narrative Arc
- Does the Part tell a coherent story from its first chapter to its last?
- Does each chapter build on the previous one appropriately?
- Are there gaps where a concept is used before it's introduced?
- Are there unnecessary repetitions across chapters?
- Does the Part introduction (foundations_principles.qmd, etc.) accurately
  preview the chapters that follow?

## B. Terminology Consistency
- Are key terms defined the same way across all chapters in this Part?
- If a term is redefined or refined, is the evolution explicit?
- Are acronyms expanded on first use within each chapter?

## C. Cross-Reference Resolution
- Collect all @sec-*, @fig-*, @tbl-*, @eq-* references in each chapter.
- Verify that references WITHIN this Part point to sections that exist.
- For references OUTSIDE this Part, verify the target label exists in the
  target chapter (read the target chapter's section headers).
- Flag any broken or suspicious cross-references.

## D. Numerical Consistency
- When multiple chapters cite the same hardware spec (e.g., "A100 has 312 TFLOPS"),
  verify they all use the same number.
- When one chapter says "as we saw in Chapter X, the value was Y", verify
  Chapter X actually states Y.
- Verify that all chapters source their numbers from constants.py rather than
  having hardcoded values that could drift.

## E. Concept Dependencies
- Map the concept dependency graph: what concepts does each chapter assume
  the reader already knows?
- Are there circular dependencies?
- Are there concepts that should be introduced earlier but aren't?

## F. Transition Quality
- How does each chapter end? Does it set up the next chapter?
- How does each chapter begin? Does it connect back to what came before?
- Are the "What's Next" callouts (callout-chapter-connection) accurate?

OUTPUT FORMAT:
- For each check (A-F), list findings with specific file:line references.
- End with a prioritized list of issues to fix before press.
```

---

#### Part I: Foundations (4 chapters)

```
PART: I — Foundations of ML Systems

CHAPTERS TO READ (in order):
1. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/parts/foundations_principles.qmd
2. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/introduction/introduction.qmd
3. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/ml_systems/ml_systems.qmd
4. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/workflow/workflow.qmd
5. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/data_engineering/data_engineering.qmd

SPECIAL ATTENTION:
- The D·A·M taxonomy introduced in Chapter 1 must be consistently referenced.
- Chapter 2 (ML Systems) should feel like a natural deepening of Chapter 1.
- Chapter 3 (Workflow) should transition from "what" to "how".
- Chapter 4 (Data Engineering) should be clearly positioned as the data
  foundation that Part II will build on.
- Verify the Part introduction accurately frames these four chapters.
```

#### Part II: Development and Training (4 chapters)

```
PART: II — Development and Training

CHAPTERS TO READ (in order):
1. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/parts/build_principles.qmd
2. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/dl_primer/dl_primer.qmd
3. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/dnn_architectures/dnn_architectures.qmd
4. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/frameworks/frameworks.qmd
5. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/training/training.qmd

SPECIAL ATTENTION:
- DL Primer must establish math foundations used by subsequent chapters.
- Architectures chapter should reference DL Primer math (not re-derive it).
- Frameworks chapter should use architectures from the previous chapter as
  implementation examples.
- Training chapter should build on all three preceding chapters.
- Verify parameter count / FLOPs numbers are consistent across DNN Architectures,
  Frameworks, and Training chapters.
- Check that any code examples in Frameworks use architectures defined in the
  DNN Architectures chapter.
```

#### Part III: Optimization and Acceleration (4 chapters)

```
PART: III — Optimization and Acceleration

CHAPTERS TO READ (in order):
1. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/parts/optimize_principles.qmd
2. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/data_selection/data_selection.qmd
3. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/optimizations/model_compression.qmd
4. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd
5. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/benchmarking/benchmarking.qmd

SPECIAL ATTENTION:
- This Part maps to the D·A·M axes: Data Selection (D), Model Compression (A),
  Hardware Acceleration (M), Benchmarking (cross-cutting).
- Verify this D·A·M mapping is explicit and consistent.
- Model Compression and Hardware Acceleration are tightly coupled — verify
  cross-references between them are accurate (e.g., "quantization to INT8
  enables tensor core acceleration").
- Benchmarking should reference techniques from all three preceding chapters.
- This Part has the heaviest calculation load — verify that physx calculations
  are consistent across all four chapters.
- Check that hardware specs (A100, H100, etc.) are identical in Model
  Compression and Hardware Acceleration chapters.
```

#### Part IV: Deployment and Operations (4 chapters)

```
PART: IV — Deployment and Operations

CHAPTERS TO READ (in order):
1. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/parts/deploy_principles.qmd
2. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/serving/serving.qmd
3. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/ops/ops.qmd
4. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd
5. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/conclusion/conclusion.qmd

SPECIAL ATTENTION:
- Serving should connect back to optimization techniques from Part III.
- Ops should build on serving infrastructure.
- Responsible Engineering should weave together concerns from ALL prior Parts.
- Conclusion should tie back to the D·A·M framework from Chapter 1.
- Verify the conclusion accurately summarizes all 4 Parts.
- Check that the conclusion's preview of Volume II is appropriate.
- Verify any claims about production deployment numbers/statistics.
```

---

## Phase 3 (Optional): Full-Book Structural Audit

Run this as a single agent after Phases 1 and 2.

```
You are performing a final structural audit of "Introduction to Machine Learning
Systems" (Volume I) for MIT Press.

READ THESE FILES:
1. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/config/_quarto-pdf-vol1.yml (full file)
2. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/parts/summaries.yml
3. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/index.qmd
4. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/frontmatter/notation.qmd
5. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/contents/vol1/backmatter/glossary/glossary.qmd
6. /Users/VJ/GitHub/mlsysbook-vols/book/quarto/tex/header-includes.tex

Then SCAN (read first 50 lines + last 20 lines of) every chapter .qmd file.

CHECK:
1. Chapter ordering in _quarto-pdf-vol1.yml matches the logical flow.
2. Every chapter listed in the config actually exists on disk.
3. Every .qmd file in the vol1 directory tree is included in the config
   (no orphaned files).
4. The notation guide covers all symbols used across chapters.
5. All key terms from chapters appear in the glossary.
6. The index entries (\index{}) are well-organized:
   - No duplicate entries with different casing
   - Subentries are used appropriately (Term!subterm)
   - Important terms aren't missing index entries
7. The bibliography (references.bib) is well-formed and all cited references
   exist.
8. The abstract in the config accurately describes the book's content.
9. LaTeX header-includes.tex doesn't have any issues that could affect rendering.

OUTPUT: A prioritized punch list of issues to fix before submitting to press.
```

---

## How to Run These Prompts

### In Cursor (recommended)

Use the Task tool with `subagent_type="generalPurpose"` for each prompt.
Run up to 4 agents in parallel per batch:

**Batch 1:** Chapters 1-4 (Part I)
**Batch 2:** Chapters 5-8 (Part II)
**Batch 3:** Chapters 9-12 (Part III)
**Batch 4:** Chapters 13-16 + Conclusion (Part IV)
**Batch 5:** Appendices A1-A5
**Batch 6:** Phase 2 cross-chapter reviews (all 4 Parts)
**Batch 7:** Phase 3 structural audit

### Estimated Token Usage

Each chapter agent will read:
- ~3 shared context files (~1500 lines)
- ~1 chapter file (1000-5700 lines)
- ~1-2 calculation files (~400 lines)

Total input per agent: ~5K-15K lines = ~20K-60K tokens
Total for all 25 agents: ~500K-1M input tokens

### Collecting Results

Each agent outputs a structured report. Collect all reports and:
1. Triage by severity (Critical > Warning > Minor)
2. Fix all Critical issues first
3. Cross-reference Phase 1 outbound references with Phase 2 findings
4. Use Phase 3 as final verification
