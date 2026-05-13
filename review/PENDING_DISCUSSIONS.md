# Pending discussions per chapter

Items where I made a judgment call against the YAML, kept something the YAML flagged, or saw something the YAML missed. We can walk through these in the batch session.

## frameworks.qmd

- **Kept the five-question abstraction-problem scaffold** (lines 2945, 3263, 3546, 3629). YAML didn't flag this directly, but it's the kind of pedagogical pattern that strict rule-reading would convert. I left it because the questions are *the framework's* analytical tasks (italicized, ordinal-referenced) and form the organizing backbone for ~700 lines. Already codified as the "pedagogical scaffold exception" in book-prose.md.

## nn_architectures.qmd

- **Kept seven italicized organizing questions** (lines 58, 195, 237, 875, 1503, 1675, 1747). The chapter's golden thread is the recurring italicized thesis-question *"how should we structure computation to match the structure in our data?"* — a chapter-spanning organizing motif, not reader-address. Same exception as frameworks.
- **Kept callout "Problem:" question openers** in worked-example callouts at lines 1201, 1766. Standard pedagogical scenario-opener pattern.

## nn_computation.qmd

- **BurBank → Burbank**: standardized caption to match alt text. The YAML asked to verify against the source image, which I couldn't do. If the real USPS mail sample is mixed-case "BurBank" (the chapter preserves "TULSA" all-caps for fidelity), the caption should be reverted. **Action needed**: confirm against the actual figure image.
- **post-processing vs. postprocessing**: standardized on "post-processing" (16 instances) by replacing the 3 "postprocessing" outliers. The book-prose rule §10.8 closes up `pre-` (pretraining, preprocessing, etc.) but is silent on `post-`. Worth deciding the house form for `post-` and adding to §10.8 — current standardization assumes hyphenated.

## training.qmd

- **`.callout-example` "The 3AM gradient explosion" structure** (line 168): currently uses Scenario/Failure/Physics/Fix labels (with "The Physics" title-case fixed to "The physics"). Book-prose §4 specifies `.callout-example` should use Context/Insight/Lesson, and `.callout-war-story` should use Context/Failure/Consequence/Systems-lesson. This callout describes a production failure with quantified consequence and fix — closer to a war-story than an example. **Action needed**: change callout type to `.callout-war-story` and rename the four bold labels to the canonical war-story arc? I left the type unchanged for now to avoid restructuring.
- **FlashAttention-3 citation gap**: the chapter claims FA3 reaches ~740 TFLOPS on H100 (~75% of peak). No bib entry exists for the Shah et al. 2024 paper. I softened to "the authors report" framing. **Action needed**: add a `dao2024flashattention3` (or `shah2024flashattention3`) bib entry and switch back to a direct cite.
- **GPT-2 lighthouse ambiguity**: I labeled the optimizer callout as "GPT-2 XL training configuration" and the 50K-step convergence claim as "this illustrative configuration." If the chapter's lighthouse-model framing in @sec-introduction calls it just "GPT-2" without an XL qualifier, this naming may need to be re-aligned.

## data_selection.qmd

- **Blocker (YAML) at line 3286**: the YAML reported a missing closing backtick before `)` in the ROI inline-Python expression. Inspecting the raw line shows all four backticks present and balanced. The blocker appears already fixed or the YAML was over-reading. **Action needed**: render the chapter and confirm.
- **Chinchilla rewrite**: the YAML flagged $D_{\text{opt}} \propto C^{0.5}$ paired with "40 percent more data" when doubling compute. I rewrote to make the simplifying assumption explicit (model size held proportional to its compute-optimal allocation; tokens then grow as $\sqrt{C}$ giving ~41% per doubling), and added a sentence linking to the underlying Chinchilla tokens-parameter balance. **Action needed**: confirm the simplification is the framing the chapter wants, or replace with a coupled-allocation derivation.
- **Figure annotation fix at @fig-ppd-curve**: changed the misleading matplotlib label `"Efficiency Gap (Saved Compute)"` to `"Performance Gap (Same Data Budget)"`. The vertical arrow shows accuracy gained at fixed data, not compute saved. Caption + alt updated. **Action needed**: re-render and confirm the new label reads correctly at print scale.

## model_compression.qmd

- **Figure source attribution gaps** at lines 2984, 4508, 5721, 7191: captions cite "Source: IEEE Spectrum", "Source: HarvardX", "Source: NVIDIA", "Source: PyTorch Documentation" without bib keys. Per bib-check policy, this is bib work, not prose work. **Action needed**: add proper bib entries or convert to "adapted from" with cited sources.
- **Fallacies-section empirical claims uncited** at line 7893: specific BERT/ResNet-50/INT4/binary weight numbers in the fallacies section read as measured literature results without inline citations. **Action needed**: cite each empirical claim or mark as illustrative scenarios derived from chapter constants.
- **Technique-selection table fix at @tbl-constraint-opt-mapping**: changed "Latency and Throughput × Numerical Precision" from ✗ to △ (hardware-dependent) and added a caption sentence explaining the hardware dependency. The earlier ✗ contradicted the chapter's main quantization-as-latency-optimization framing.

## hw_acceleration.qmd

- **Subsection move at @fig-memory-wall**: moved the `#### Irregular memory access` heading + its index + intro paragraph DOWN past the figure block. Figure now belongs to the memory-wall parent section as the YAML requested. **Action needed**: re-render and confirm cross-reference @fig-memory-wall still resolves and the irregular-access subsection still reads cleanly with its own intro.
- **Activation-stationary → Input-stationary in @tbl-mapping-strategies**: chapter defines "Input Stationary" as the third dataflow strategy, but the table used "Activation Stationary". Changed to "Input Stationary" and added a parenthetical explaining the KV-cache reuse case.
- **NHWC/NCHW table cells in @tbl-mapping-strategies**: changed from fixed NCHW/NHWC prescriptions to "Backend-dependent" with kernel-path examples, matching the earlier backend-dependent layout caveat. Also rewrote line 5112 to make the cuDNN/Tensor Cores/TPU XLA distinction explicit. **Action needed**: confirm the revised wording matches the chapter's later mapping examples and doesn't break any cross-reference.
- **Intel Sapphire Rapids "tensor cores" → AMX tile engines**: corrected vendor terminology (line 2280).
- **Three trailing section IDs removed** at lines 5070, 5097, 5116, 5137: `{#sec-hardware-acceleration-*-importance-*}` anchors at the end of paragraphs. Verified none were referenced elsewhere. Removed.
- **`\text{four}^{5 \times 3}` → `4^{5 \times 3}` math fix** (line 4233): malformed text-in-exponent corrected.
- **"ultimate Green technology" promotional claim** softened to a workload-specific energy reduction claim (line 5755).

## benchmarking.qmd

- **Pricing data citations not added** (lines 197, 4228): the intelligence-deflation figure and cloud-instance cost table use time-sensitive prices without dated sources. Per bib-check policy, this is bib work, not prose work. **Action needed**: add citations or replace "as of this writing" with a specific date + source.
- **Power-range narrative reconciled** with @tbl-power: rewrote the caption to explicitly distinguish the table's 150 µW–10 kW span (seven orders of magnitude) from the figure's 5.6 µW–498 kW span (closer to ten orders). Both numbers stand; the caption now bridges them.
- **Kept the three italicized organizing questions at line 106** (system / model / data benchmarking dimensions) as a chapter-organizing pedagogical scaffold. Same exception as frameworks and nn_architectures.
- **Kept the "Goodhart's Law in action" callout** structure (Problem / Math / Result / Systems-conclusion). Removed the "Better!", "The moral", "AI Engineering", "won the leaderboard but destroyed the product" colloquial phrasing per YAML.

## model_serving.qmd

- **Inline-Python-in-math at lines 2363, 2373, 2379**: converted the `\(...\{python}...\)` and `$$...{python}...$$` patterns to plain prose with inline-Python references outside math delimiters, since `$$...$$` does not support backtick-Python evaluation. The math now renders reliably in both HTML and PDF.
- **Batching-window formula at line 3419** **[REVISIT CAREFULLY]**: relabeled as a heuristic rather than an "optimum"; verified dimensional consistency (S/λ has units of seconds² so √ gives seconds) and added explicit unit annotations. Also softened "$T_{\text{optimal}}$" to "$T_{\text{window}}$" since no derivation is provided. **Action needed**: focused session to decide between (a) keep `T_window` heuristic framing or (b) restore `T_optimal` with a published derivation/citation (e.g., M/M/1 cost-balance result, or a specific batching paper). User flagged this needs careful treatment.
- **DLRM/ResNet transition fix at line 1725**: rewrote the post-DLRM-callout transition to acknowledge both lighthouses rather than reverting to ResNet image-decoding.
- **Llama-3-8B case study disambiguation at line 4254**: added an explicit framing paragraph noting that @fig-kv-cache-growth uses 70B-class assumptions as a contrast before the 8B-specific analysis. **Action needed**: consider regenerating the figure with 8B-class assumptions if the chapter wants strict scope alignment.
- **Pricing citations** at lines 197, 4228: same as benchmarking — out of scope for prose pass.

## ml_ops.qmd

- **Retraining-cost equation at line 1417**: the YAML reported unbalanced `\text{Retraining Cost}` braces but the current state shows balanced TeX. Either already fixed or YAML over-read. **Action needed**: render and confirm.
- **Multiple citation gaps** at line 147 (definition callout 10–20% / KL threshold / MTTR), line 153 (retail failure: 15% boost / 5% loss / USD 10M), and others: presented as unsupported precise claims. Per bib-check policy, these need bib entries. **Action needed**: cite each, qualify as illustrative, or rewrite as a hypothetical scenario.
- **Prose double-hyphens in footnotes** at lines 333, 506, 676, 777, 825, 841, 884, 1179, 1183, 1249, 1313, 1582, 1598, 1600, 2068, 2072: I fixed only the most visible (lines 94 and a few ranges). The remaining footnote `--` instances are still present. **Action needed**: sweep the remaining footnotes or batch-convert via a follow-up pass.

## responsible_engr.qmd

- **Fairness-archetype table at line 678** rewritten as four single-row entries with the archetype label folded into one cell. The original two-row "Compute Beast / Bandwidth Hog / Sparse Scatter / Tiny Constraint" pattern was relying on implicit rowspans that don't render in Markdown tables. New form has wider cells but accurate row structure.
- **Fairness-frontier figure (line 249) caption-vs-axis mismatch**: I did NOT verify against the rendered figure. The YAML flagged that `ax.invert_xaxis()` puts low-disparity on the right while caption says Point C is on the "left/low-disparity" side. **Action needed**: render the figure, check axis orientation, and fix either the caption or the `invert_xaxis()` call.
- **"Price of fairness" 3% vs 4.8% mismatch at line 916**: I did NOT fix this. The Python cell hard-codes a 3% utility loss with comments mentioning ~4.8% and 30% group-share, making the result non-reproducible from visible assumptions. **Action needed**: parameterize group_proportion and let the computed value drive the prose figure.
- **USD k-suffix formatting at line 960**: I did NOT fix this. `hire_value_k_str = "USD 100k"` mixes informal `k` suffix with formal "USD" prefix. **Action needed**: change to `USD 100,000` for consistency with chapter elsewhere.

## conclusion.qmd

- **"twelve" anchor in heading "Thirteen Quantitative Invariants"**: I deliberately did NOT rename `#sec-twelve-quantitative-invariants-0dd2` and `#tbl-twelve-principles`. Both are referenced by `ml_ops.qmd:72`, `ml_ops.qmd:1406`, and `conclusion_quizzes.json`. Renaming would break cross-references. The reader-visible heading and table caption already say "thirteen". **Action needed**: decide whether to (a) keep the stable anchors and accept the maintainer-confusion, (b) rename the anchors and update all four cross-references in lockstep, or (c) add a `{#sec-thirteen-...}` alias next to the existing anchor.

## Status: vol1 prose pass complete

All 26 chapters in `_quarto-pdf-vol1.yml` order have been touched. The next step is the batch interactive session to resolve the items above.

---

# Vol2 prose pass

## vol2/frontmatter/about.qmd

- **Part-introduction italicized questions (lines 37, 41, 45, 49)**: Each Part is introduced as `Part X: Name --- *italicized organizing question?*` followed by a descriptive paragraph. Vol1's about.qmd uses a declarative bullet form (`**Part I: Foundations** develops...`). **Options**: (a) keep the italic-Q framing (textbook-pitch style, distinctive); (b) convert to vol1's declarative form (consistency across volumes); (c) leave the italic-Q opener and add a brief declarative scope sentence alongside. The italic-Qs read as Part subtitles, not body-prose rhetorical questions, but the pedagogical-scaffold exception's "ordinal back-reference" criterion is not satisfied.
- **Orchestra analogy "not simply many musicians playing at once" (line 27)**: Uses the "not just X but Y" anti-pattern. Functional but flagged. **Options**: (a) leave (orchestra metaphor is well-developed and the contrast structure earns it); (b) recast as "an orchestra is qualitatively different from many musicians playing at once".
- **Uncited quantitative claims**: AlexNet "five to six days on two GPUs" (line 11), GPT-4 "25,000 GPUs running for roughly three months" (line 11), "10,000-accelerator cluster with a 2 percent annual failure rate" (line 13). **Action needed**: add `[@krizhevsky2012imagenet]` for AlexNet, soften GPT-4 numbers or cite a press estimate, mark failure-rate as illustrative or cite SDC literature (`[@dixit2021silent]`, `[@hochschild2021cores]`).
- **Fleet Stack LaTeX figures (lines 57, 67)**: Raw `{=latex}` blocks with no HTML equivalent, no alt-text, no figure label. **Action needed**: convert to cross-format figure pattern with `fig-cap`, `fig-alt`, and HTML/PDF rendering equivalence.
- **Suggested Reading Paths (lines 99–105)**: Four paragraphs with repeated "The X Path" openers. **Options**: (a) keep paragraph form (current reading is fine); (b) convert to compact list/table for scannable navigation.

## vol2/frontmatter/acknowledgements.qmd

- **"someone out there cares" (line 31)**: YAML flagged as colloquial. Acknowledgements register is intentionally personal; I left it. Confirm.

## vol2/index.qmd

- **HTML landing page register**: Sections "Support Our Mission" (line 77), "Listen to the AI Podcast" (line 117), "Want to Help Out?" (line 131) read as web/marketing copy. **Action needed**: decide whether these are intentional public-landing-page elements (keep as-is) or should match textbook register. The "Author's Note" PDF version (lines 16–34) is textbook-register and well-written.
- **`date: today` (line 5)**: Build metadata uses dynamic date. **Options**: (a) keep dynamic for development; (b) pin to release date for publishing.
- **"📖 Click here to download PDF" (line 53)**: Informal CTA. Landing-page register choice.

## vol2/frontmatter/_notation_distributed.qmd

- **Forward-reference of symbols**: The opening paragraph notes that some symbols are reserved for forward use. This is fine but should pair with anchored cross-references in the chapters that consume them (no action this pass).

## vol2/conclusion/conclusion.qmd

**Walkthrough complete.** Earlier prose-pass items resolved. Remaining 7 items consolidated into a `<!-- TODO(focused-followup) -->` block at the top of the chapter for a focused style-cleanup + numerical-audit pass.

## vol2/responsible_ai/responsible_ai.qmd

**Walkthrough complete.** Earlier prose-pass items resolved. Raw `\ref{pri-...}` cross-references at lines 399 and 2113 are **correct** per the vol1+vol2 principle-reference convention (previously flagged for conversion, but should stay). Remaining 10 items consolidated into a `<!-- TODO(focused-followup) -->` block at the top of the chapter.

## vol2/sustainable_ai/sustainable_ai.qmd

**Walkthrough complete.** Earlier prose-pass items resolved. Remaining 7 items consolidated into a `<!-- TODO(focused-followup + numerical-audit) -->` block at the top of the chapter.

## vol2/robust_ai/robust_ai.qmd

**Walkthrough complete.** Earlier prose-pass items resolved. Remaining 7 items consolidated into a `<!-- TODO(focused-followup) -->` block at the top of the chapter.

## vol2/security_privacy/security_privacy.qmd

**Walkthrough complete.** Earlier prose-pass items resolved. Remaining 5 items consolidated into a `<!-- TODO(focused-followup) -->` block at the top of the chapter.

Heading-case hook conflict items tracked separately as a lockstep slug-rename pass: `Jeep cherokee hack` (837), `Rnyi differential privacy and composition` (3063), `Meta grand teton` (network_fabrics:1360), `Advanced slurm configuration for ML` (fleet_orchestration:589). Each requires renaming the section-ID slug and the visible text in lockstep.

## vol2/ops_scale/ops_scale.qmd

**Walkthrough complete.** Earlier prose-pass items resolved. Remaining 10 items consolidated into a `<!-- TODO(focused-followup) -->` block at the top of the chapter.

## vol2/edge_intelligence/edge_intelligence.qmd

**Walkthrough complete.** Earlier prose-pass items resolved. Remaining 9 items consolidated into a `<!-- TODO(focused-followup) -->` block at the top of the chapter.

## vol2/inference/inference.qmd

**Walkthrough complete.** Earlier prose-pass items resolved. The line ~1053 plaintext-label cluster ("System Parameters", "Service Time Model", "Optimal Batch Size Calculation") was converted inline to bold lead-ins. Remaining 7 items consolidated into a `<!-- TODO(style-cleanup) -->` block at the top of the chapter.

## vol2/performance_engineering/performance_engineering.qmd

**Walkthrough complete.** Earlier prose-pass items + four deferred items resolved:

- **`fig-shifting-roofline` 4x vs 2x mismatch**: figure annotation "Ridge point shifted 4.0x in 7 years" → "~2x in 7 years" matching the 139→281 FLOP/byte progression in the prose and table.
- **`fig-shifting-roofline` duplicate plotting**: deleted the first redundant `for name, peak_tflops, bw_tbs, color in gpus:` loop; the second loop (with ridge labels) is retained.
- **Speculative Decoding stub at line ~1370**: added local pedagogical content explaining the arithmetic-intensity shift, the acceptance-rate bound, and the interaction with batching, before forwarding to `@sec-inference-scale-speculative-decoding-c438`.
- **MFU cross-ref**: corrected `@sec-performance-engineering-efficiency-frontier` → `@sec-performance-engineering-measurement-scale` (the actual measurement-at-scale section).

## vol2/fleet_orchestration/fleet_orchestration.qmd

**Walkthrough complete.** Earlier prose-pass items + deferred items resolved:

- Build-vs-buy "Is this effort justified?" rhetorical Q → declarative ("The ROI math is what justifies the effort").
- "Highest-leverage engineering investment" repetition trimmed: two of the four instances (lines 510 and 1116 region) removed; kept the strongest at line 203 (initial economic-stakes paragraph) and line 2120 (Summary takeaway).
- `2--4$\times$` range marker (line 2042): left as-is — within house style.
- 2056 three audit questions paragraph: left as-is (similar to Fallacies pattern; questions in a "checklist" context are acceptable).

## vol2/fault_tolerance/fault_tolerance.qmd

**Walkthrough complete.** Earlier prose-pass items + deferred items resolved:

- Predictive-maintenance duplicate sentence (line ~744): merged into single causal sentence.
- Repetitive "For X..." sentence frames (line ~2624): condensed into a single causal sentence about elastic training converting hard failures into graceful capacity adjustments.
- Case-study section (line ~3293): opening question removed; replaced with declarative thesis. Bib-pass TODO comment added at the start of `## Case Studies` enumerating OPT-175B/Google TPU/Netflix/DeepSpeed claims that need citations.
- 1741 notebook context (Numerical-Garbage callout): already tightened in the earlier prose pass; line numbers shifted, no fresh second-person site remains.

## vol2/collective_communication/collective_communication.qmd

**Walkthrough complete.** Earlier prose-pass items + deferred items resolved:

- BF16 70B AllReduce: `L_lat` → "bandwidth (data-movement) term" (correctly identifying the 140 GB / 50 GB/s transfer as bandwidth-bound).
- Raw `\ref{pri-alpha-beta-model}` (line ~602): **kept as `\ref{}`** per the vol1+vol2 principle-reference convention (user correction during walkthrough).
- NCCL reality-gap table caption (line ~613): relabeled as illustrative reference numbers ("Values are illustrative reference numbers... actual measurements vary with topology, NCCL version, and tuning").
- "Can an algorithm achieve the best of both?" rhetorical Q (line ~950): replaced with declarative ("A third approach combines the logarithmic latency of Tree with better bandwidth utilization").
- "Roads are constantly crumbling" colloquial chapter-connection (line ~2159): replaced with neutral "In production, however, hardware fails as a statistical certainty...".
- Compression-payback comment drift (line ~2082): code comment updated to match the actual ~5.7-fold value the cell computes.

Deferred to focused follow-up:

- **BF16 vs. FP32 precision drift for 70B running example** (lines 59/69 vs. 157/511 vs. 592): chapter alternates between BF16 (140 GB) and FP32 (280 GB) for the same model. Needs explicit precision declaration at each scenario switch, or standardization on one canonical precision.
- **Fallacies bold paragraph starters** (line ~2031): kept as canonical `**Fallacy**: *italic.* / Corrective paragraph` form per book-prose §3.
- **Compression-payback notebook after Summary opens** (line ~2073): pedagogically defensible as a final illustration before takeaways; can move into the compression-trade-offs section if a stricter Summary boundary is wanted.

## vol2/distributed_training/distributed_training.qmd

**Walkthrough complete.** Earlier prose-pass items + deferred items resolved:

- "Exponentially with cluster size" (line 160): split into the two distinct quantities — expected failures per unit time grows linearly with cluster size, while probability of full-cluster survival decays exponentially.
- Loop-transform diagram in `{text}` block (line ~225): converted to a proper `{mermaid}` fence inside a `::: {#fig-loop-transforms fig-env=...}` div with proper `fig-cap` and `fig-alt`.
- Slow-AllReduce debugging arithmetic (line ~648): TODO(numerical-audit) comment added describing the 50/100/120/240 ms inconsistency; needs careful recomputation against one consistent baseline.
- Footnote `--` parenthetical breaks (lines 261, 263, 698): converted to proper punctuation (comma clauses, colons, parens).
- 3D-parallelism summary figures reused same asset (lines 2618/2627): consolidated into a single `@fig-3d-parallelism-cube-summary` div with a merged caption that covers both the 3D strategy space and the per-step coordination story. Deleted the duplicate `@fig-3d-parallelism-sliced` figure and its surrounding cross-reference paragraph.
- Manual bold caption outside `fig-cap` (line ~2621): removed; the content moved into the merged figure caption.

## vol2/data_storage/data_storage.qmd

**Walkthrough complete.** Earlier prose-pass + deferred items resolved:

- **`@fig-access-patterns-vol2` figure-vs-caption mismatch**: Revised the surrounding prose at lines 415/423–425 to discuss the displayed I/O-throughput-vs-model-size figure (storage-bottleneck regime) instead of the sequential-vs-random comparison the prose previously described. The figure stays; the prose now matches it.
- **Retrieval-infrastructure subsection reorganization**: Moved out of `## Fallacies and Pitfalls` into its own top-level `## Retrieval Infrastructure` section before Fallacies. Section ID `#sec-data-storage-retrieval-infrastructure` preserved.
- **Numerical inconsistencies (7 sites)**: Single TODO(numerical-audit) block added at the top of the chapter, enumerating: HBM 300,000x gap, storage pyramid caption vs table, archive 5x vs 20x, 2-year cost \$960 vs \$9,600, idle-cost \$9,600 vs \$13,000, ZeRO-3 70B shard math, 175B 256-node 4 GB vs 6.8 GB. Each needs careful derivation against the running examples; queued for a focused numerical-audit pass.

## vol2/network_fabrics/network_fabrics.qmd

**Walkthrough complete.** Earlier prose-pass items + deferred items resolved:

- **Five-level taxonomy mismatch**: Figure caption labels updated to match section-heading taxonomy (Level 1 Wire and Link, Level 2 Transport, Level 3 Switch and Topology, Level 4 Fabric Behavior, Level 5 Cluster Design).
- **Oversubscription dollar discrepancy**: Line 963 hardcoded \$75M → \$142M (matching the BisectionBottleneck computed model in @sec-network-fabrics-fat-tree). Forward-reference using `{python}` not feasible without moving the LEGO cell up; literal value with explicit cross-reference is the pragmatic fix.
- **Checkpoint over-explained**: Prompt at line 1530 trimmed so it asks the diagnostic question without giving the reasoning. The reader has to identify the communication-vs-compute distinction independently.
- **Fat-tree scaling claims**: Added `<!-- TODO(numerical-audit) -->` comment before the Fat-Tree callout marking the five-site reconciliation work. Defer to a focused numerical-audit pass.
- **Fallacies bold paragraph starters**: After review, the `**Fallacy**: *italic statement.* / Corrective paragraph` format is the canonical book pattern explicitly specified in book-prose.md §3 (Fallacies and Pitfalls). YAML's flag was inconsistent with house style. Kept.

Queued for focused follow-up:

- **Bandwidth-staircase figure (line ~1350)**: `@fig-hierarchical-staircase` rendered asset doesn't match prose intent. Needs either new figure asset or prose revision.
- **Rail-optimization data-vs-tensor confusion (line ~998)**: Mixes traffic patterns. Needs technical clarification separating same-rank data-parallel AllReduce from tensor-parallel activation exchange.

## vol2/compute_infrastructure/compute_infrastructure.qmd

**Walkthrough complete.** Earlier prose-pass items + five deferred items resolved:

- **Roofline ridge precision standardized on FP16 (~295 FLOP/byte)**: Footnote line 825 was already correct (989 / 3.35 ≈ 295). Fixed HBM3 definition callout at line 517 (was using A100's 312 TFLOPS — corrected to H100's 989 TFLOPS). Fixed figure caption (~591 → ~295 FP16). Updated `H100_FLOPS_FP8_TENSOR` → `H100_FLOPS_FP16_TENSOR` in the roofline figure LEGO cell and the roofline-analysis LEGO cell. Adjusted prose "591× gap" → "~2,000× gap" (the actual intensity ratio).
- **Cooling rack count**: "128-rack training cluster" → "32-rack training cluster" at line 2225 (matches the 1,024 GPU / 32-rack running example used elsewhere).
- **TCO figure caption**: "1,000-node H100 cluster (8,000 GPUs)" → "1,000-GPU H100 cluster (125 nodes at 8 GPUs/node)". Cloud cost was already correct for the smaller scale; caption now matches surrounding prose.
- **Bold Step starters in planning-methodology example (line 3067)**: Converted seven `**Step N**:` paragraph starters into flowing prose with topic sentences. Also propagated the FP16 precision fix into the worked-example math ($1,979 \to 989$ TFLOPS, $891 \to 445$ TFLOPS sustained, $912 \to 456$ PFLOPS cluster). One-week idealized compute time stays in the 2–4 week operational-overhead range.

Queued for bib pass:

- Environmental-impact citations (line ~2165): GPT-3 energy estimate, carbon-intensity values, Microsoft nuclear procurement, Google geothermal investment.

## vol2/introduction/introduction.qmd

**Walkthrough complete (2026-05-11).** All nine YAML structural items resolved:

- **C³ taxonomy figure dedup**: SVG kept as canonical at `@fig-c3-taxonomy` (the original ID retained); TikZ source **deleted** (git history preserves it). Duplicate `@fig-vol2-c3-taxonomy` block deleted; line 1665 prose redirected to `@fig-c3-taxonomy`.
- **Lighthouse archetypes**: Early callout at line 328 region **kept** (vol2 is independent of vol1; readers need the introduction at first mention) and edited to add a forward pointer to `@sec-vol2-introduction-archetypes` for the canonical roster. The early callout serves as the introduction; the section + table at line 1912 is the deeper reference.
- **Cloud Archetype vs Archetype A naming**: Line 806 rewritten to use the canonical "Archetype A (GPT-4 / Llama-3)" label with `@sec-vol2-introduction-archetypes` reference.
- **C³ corner mapping**: Archetypes now cleanly map to C³ corners — A=Communication (fleet-wide), B=Coordination (all-to-all), C=Compute (per-device envelope). Prose at line 1910 and table at line 1916 aligned to use C³ vocabulary; "Throughput Bound / Volume & Latency / Power & Privacy" replaced with the corner labels. Archetype C's "Compute" framing clarifies that the binding constraint is the per-device milliwatt envelope, not fleet-level network or coordination.
- **Scale-Moment magnitude**: Line 103 tightened to "approaching $10^{25}$ FLOPs ... roughly seven orders of magnitude" — matches table caption.
- **Foundational Concepts section**: On inspection, this section is the **first introduction** of AI Triad and Five-Pillar Framework (not a re-introduction); the only re-introduced framework is C³, and that was already resolved by the figure dedup. Fleet Stack figure here is the canonical visual. Three systems archetypes here is the canonical roster. No further edit needed.
- **Purpose paragraph density (line 28)**: Kept as-is — manifesto register intentional for volume opener.
- **Fleet Stack + Roadmap figure duplication**: Kept both — Fleet Stack figure shows the layered architecture; Roadmap figure shows chapter sequence through the same four parts. Distinct pedagogical purposes.
- **Six Systems Engineering Principles bullets**: Kept as compact reference list.
