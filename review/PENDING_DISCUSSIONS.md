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
- **Batching-window formula at line 3419**: relabeled as a heuristic rather than an "optimum"; verified dimensional consistency (S/λ has units of seconds² so √ gives seconds) and added explicit unit annotations. Also softened "$T_{\text{optimal}}$" to "$T_{\text{window}}$" since no derivation is provided. **Action needed**: if the original `sqrt(S/λ)` formula has a published derivation, restore "$T_{\text{optimal}}$" and cite the source.
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

Applied: `high bandwidth flash` → `high-bandwidth flash` compound-modifier hyphen (241), brain-synaptic-activity rhetorical Q recast and "intuition pump" colloquial replaced with "Fermi-style sanity check" (359), "We have conquered Scale" triumphalist phrasing softened (378).

Deferred to discussion:

- **Compound capability law callout (177)**: bold paragraph starters "Observation/Principle/Implication" + rhetorical Q.
- **Capability ∝ Model_IQ × Tools × Context × Planning^N speculative formula (184)**: define as heuristic or add derivation.
- **Notebook callout rhetorical Qs (226, 277)**: convert.
- **`1,000 TFLOPS` hardcoded vs `H100_FLOPS_FP16_TENSOR=989` (364)**: use computed value or mark approximate.
- **Author signature before takeaways (388)**: move signature after takeaways or remove from chapter body.
- **Time-sensitive phrases "today's challenges" / "next decade" / "already here" (173)**.
- **"Build systems that..." manifesto-style closing (386)**.

## vol2/responsible_ai/responsible_ai.qmd

Applied: "conscience"/"marvels" anthropomorphic opening recast (77), "You now possess..." second-person + rhetorical Q recast (79), CI memory-leak vs fairness-regression rhetorical Q → declarative pair (133).

Deferred to discussion:

- **Non-protected callout direct address (125, 281, 1098, 1522)**: multiple sites with "Use the section structure..." / "your immediate needs".
- **Navigation callout subject-verb disagreement (117)**: "Principles and Foundations ... defines".
- **Raw `\ref{...}` (399, 2113)**: convert to Quarto cross-ref.
- **SHAP overhead 50--200 percent vs 50--1000x vs 200ms–5s+ (1044)**: normalize across exact SHAP / TreeSHAP / approx / async paths.
- **Listing two adjacent string literals (1590)**: combine into single docstring.
- **`confusion_matrix(...).ravel()` without `labels=[0,1]` (1633)**: edge case fails.
- **Missing apostrophes throughout (190, 929, 1717, 2131)**: "the models", "a users", "an individuals", "the systems".
- **`bias-loop.svg` reused for `fig-bias-amplification` (907) and `fig-bias-loop` (2117)**.
- **FCC rural-broadband statistic uncited (865)**.
- **Hyphen ranges in decision-framework table (2427)**.
- **Compound forms drift (180): datacenter/data-center, high-traffic, real-time, post hoc/post-hoc, group-specific**.

## vol2/sustainable_ai/sustainable_ai.qmd

Applied: fleet-stack callout "Here, we ensure" recast + "burns through" → "exceeds" (73), operational/embodied rhetorical-Q transition → declarative (1250 region), TinyML table split row merged into single "Neural Architecture Search for MCUs" cell (2539), `CO2` → `CO~2~` updated nearby, `datacenter` → `data center`.

Deferred to discussion:

- **Operational-carbon equation PUE double-count (1256)**: `E_total` ambiguity between IT and facility-included.
- **Two near-duplicate energy-wall figures + narrative (373 vs 638)**.
- **Fallacy grid-intensity numbers vs worked example (2848)**: 367/34.5 = 10.6 not 21.
- **Embodied-carbon order-of-magnitude inconsistency (2852 vs 1501–1506)**: 108 kg amortized vs 10.5 tons upfront.
- **GPT-3 household-energy comparison drift (85/91/304/665)**: 100/120/122/130.
- **OpenAI supercomputer reference uncited/undated (1908)**.
- **"Noise Event", "Voltage Spike", etc. invented-label capitalization (653)**.

## vol2/robust_ai/robust_ai.qmd

Applied: fleet-stack callout "deep in" / "armored" / "absorb a hit" colloquial recast (53), robustness-tax Problem callout second-person → impersonal (618), `Decision Continue` plaintext-label → `**Decision**:` (1099).

Deferred to discussion:

- **`@fig-poisoning` caption-vs-prose contradiction (1866)**: caption describes online-learning incremental poisoning; prose describes Nightshade-style concept poisoning.
- **Stale listing line refs (2030)**: prose cites 2190/2196/2199-2200; listing actually at 2008-2025.
- **Attack-taxonomy table single-cell concatenation (1283)**: `FGSM PGD JSMA` etc. needs row/separator split.
- **Notebook callouts 627/1936/1945 second-person + Qs**: similar pattern, additional sites.
- **Informal figure-caption sources (1225, 2056, 2209)**: `ivezic`, `li`, `dertat`, uppercase `HTTPS://`.
- **Quantitative claims uncited (728, 744, 746)**: nighttime degradation, weather mAP, fraud drift.
- **Spaced triple-hyphens as em-dashes (83, 97)**: `---` source-style.

## vol2/security_privacy/security_privacy.qmd

Applied: DP-utility "kills utility" colloquial recast (190), `ML-allowd` → `ML-enabled` typo (1811), secure-multi-tenancy Problem callout second-person → impersonal (228), DP rhetorical-Q transition → declarative (2809).

Reverted on commit (heading-case hook conflict — section IDs encode lowercase form):

- `Jeep cherokee hack` (837): hook enforces slug = visible-text case. Reverting "Cherokee" capitalization keeps hook green. **Action**: rename slug to `cherokee→Cherokee` (lockstep) OR adjust heading-case hook proper-noun list.
- `Rnyi differential privacy and composition` (3063): same. Heading kept as `Rnyi` (no diacritic) to match slug. **Action**: rename slug to include `renyi` AND visible text to `Rényi` in lockstep.

Also reverted similarly: `Meta grand teton` (network_fabrics:1360), `Advanced slurm configuration for ML` (fleet_orchestration:589) — same hook conflict on proper-noun product names embedded in H3 slugs.

Deferred to discussion:

- **Health-monitoring Problem second-person (2446)**: similar pattern, additional notebook callout site.
- **Tesla/Zoox case study alleged-vs-established (1386)**: legal-care needed.
- **Knowledge-check second-person opener (1875)**.
- **Hardware-security mechanisms table split rows (2382)**.
- **Salary example second-person narration (2813)**.

## vol2/ops_scale/ops_scale.qmd

Applied: sharing-dividend Problem callout second-person → impersonal (152), "Even" mid-sentence cap typo fixed (2202).

Deferred to discussion:

- **Inference-cost equation dimensional analysis (line ~3265)**: `24 × 365` factor double-counts; equation as written overstates by 24x.
- **3-sigma alerts weekly/daily mismatch (line ~2193)**: same setup yields different cadences.
- **Non-protected `callout-notebook` second-person + Qs (lines 152, 326, 600, 725, 1396, 1493, 2401)**: 152 fixed; remaining sites need similar conversion.
- **"Sharing Dividend" / "Infrastructure Multiplier" / "for free" / "secret" promotional register (line ~161)**.
- **Bold paragraph starter callout (line ~3456)** + organizational-pattern bold-starter list (line ~3841).
- **Organizational-pattern figure caption left/center/right not visible (line ~3768)**.
- **Worked example after selection table redundant (line ~3933)**.
- **Case-study overclaims (Uber/Vertex AI/Netflix/Spotify, line ~3964)**: tighten citations.
- **Netflix interleaving "dramatic" intensifier (line ~3984)**.

## vol2/edge_intelligence/edge_intelligence.qmd

Applied: non-IID footnote expansion fixed (302), TikZ "loT" typo → "IoT" (597), "embraces chaos" → "tolerates" (2369), unit spacing 48MHz/3GHz/10μW (3094), memory-class hyphen ranges → en-dash (938).

Deferred to discussion:

- **`@fig-...` Local Only vs Centralized Cloud caption/prose flip (line ~308)**: A/B/C regions inconsistent between figure and prose.
- **Constraint-solution table missing Federated Coordination pillar (line ~972)**: table only covers two of three pillars.
- **Convergence-bound math F(theta) vs theta-star (line ~2255 & 2271)**: heterogeneity multiplier breaks beta=0 → IID-rate sanity check.
- **Client scheduling repetition (line ~2383 vs 2385)**: stratified sampling and adaptive client selection re-stated in adjacent paragraphs.
- **Performance-metrics paragraph (line ~2976)**: comma splices, overlong sentences. Needs prose rewrite.
- **Non-protected callout-notebook second-person + Qs (lines 160, 921, 1024, 1819)**: similar pattern, multiple sites.
- **Adaptation-strategy table caption vs row contents (line ~1605)**.
- **Worked-example plaintext label (line ~2279)**: needs callout or subsection.
- **`$E = 2$-$5$` math hyphen between math fragments (line ~2309)**.

## vol2/inference/inference.qmd

Applied: serving-cost-multiplier Problem callout second-person → impersonal (238), KV-cache wall rhetorical Q → declarative (2036), quantization opener "crush/blisteringly/brutally/massive" hype → declarative quantitative claim (4962), citation spacing fix `][@` → `] [@` (4976), case-studies hype + "we will now examine" → declarative transition (5245), cross-cutting "Separation of concerns" malformed bold-starter punctuated (5545 region), 1-5 percent hyphen → en-dash.

Deferred to discussion (most high-yield items):

- **Subsection-like labels rendered as plain paragraphs (line ~1053)**: "System Parameters", "Service Time Model", "Performance at Different Batch Sizes". Convert to proper headings.
- **Quantization-section labels rendered as plain text (line ~4991)**: "Key insight", "Algorithm", "Core technique", "Critical distinction".
- **Case-study recurring labels rendered as plain text (line ~5259)**: "Scale and requirements", "Architecture overview", "Key design decisions", "Lessons learned" — convert to consistent subheads or flowing prose.
- **Other cross-cutting bold-starter sentences (~5545 region)**: "Hybrid architectures No system..." style still needs full sentence rewrite for the others I didn't reach.
- **Fallacy "More GPU memory always means more batch size..." dangling without explanation (line ~5594)**: missing explanatory paragraph; Summary subsection appears immediately after.
- **Case-study uncited proprietary claims (line ~5257)**: Meta, OpenAI, Google, TikTok specific infrastructure numbers.
- **"4-5 requests", "10--30 percent" range form (line ~5572)**: en-dash audit.
- **Fallacies-and-Pitfalls close hype "robust / fiercely responsive / We conclude" (line ~5641)**: rewrite.

## vol2/performance_engineering/performance_engineering.qmd

Applied: stray trailing QMD fragment + duplicate `.quiz-end` removed (line 2210), outlier-features footnote `Post-Training Quantization` → `post-training quantization` (PTQ kept hyphenated per term-of-art exception), blank line restored between footnote and following body prose at line 1002, `torch.compile` heuristic "No downsides" qualified with shape-stability / graph-break caveat, scholar's-library callout second-person + "walking down there takes an eternity" recast in third person, `Hero Runs` and `Gold Standard` lowercased in body prose.

Deferred to discussion:

- **`fig-shifting-roofline` 4.0x vs 2x mismatch (line ~482)**: figure annotation says 4x, prose+table say ~2x.
- **`fig-shifting-roofline` duplicate plotting (line ~405)**: roofline series plotted twice in the LEGO cell.
- **`Speculative Decoding` near-empty section (line ~1371)**: redirects to another chapter; should add local pedagogical content.
- **MFU cross-ref to wrong target (line ~1855)**: `@sec-performance-engineering-efficiency-frontier` is the iron-law subsection, not measurement-at-scale.

## vol2/fleet_orchestration/fleet_orchestration.qmd

Applied: deadlock Problem callout second-person → impersonal (245), partition-dilemma rhetorical Qs → declarative pair (280), re-scheduling → rescheduling (298), Slurm heading capitalization (589), elastic-training transition rhetorical Q → declarative (925), cost-aware multi-question paragraph → declarative trade-off list (1316), parenthetical time-value Q → noun phrase (1318), 4-debug Problem second-person + Q → declarative (1956), `\\$` double-escape → `\$` (2008), debug-week Problem second-person → impersonal (2020), summary subject-predicate split with stray comma (2102, 2106).

Deferred to discussion:

- **2042 `2--4$\times$` range marker**: house-style debate; left as-is for now.
- **2056 three audit questions paragraph**: not yet converted.
- **1419 build-vs-buy "Is this effort justified?" inline rhetorical Q**: not yet converted.
- **Repetition of "highest-leverage engineering investment" (lines 203, 510, 1116, 2120)**: pick strongest instance, vary or delete others.

## vol2/fault_tolerance/fault_tolerance.qmd

Applied: 9s-of-reliability Problem callout second-person → impersonal (293), Byzantine fig-cap "validational redundancy" → "redundant validation" (typo fix), Numerical-Garbage colloquial recast (~line 870 region), radiation-or-beam-testing leading sentence-fragment fixed (1852), em-dash–unclosed parenthetical at storage cross-ref fixed (2322), `$k=1–3$` math en-dash range → `$1 \le k \le 3$` (2363), 1-2/0.01-0.1/1-10 hyphen ranges → en-dash, "It has been established" + "But"-start chapter-connection recast, "lesson here is" + "democratizes" recast.

Deferred to discussion:

- **1741 region notebook second-person + "Numerical Garbage" alarmist phrasing**: I tightened the line at the systems-insight callout, but the YAML may have flagged a separate second-person notebook at 1741 — re-confirm against current line numbers.
- **Predictive-maintenance duplicate sentence (line ~744)**: Two adjacent sentences say the same thing. **Action**: merge.
- **Repetitive "For X..." sentence frames (line ~2624)**: List rhythm in body prose. **Action**: condense into causal paragraph.
- **Case-study uncited operational claims (line ~3293)**: OPT-175B failures, Google TPU pod recovery targets, Netflix MTTR, DeepSpeed overhead figures — all uncited.
- **3289 rhetorical Q opening case-study section**: Already partially addressed via "It has been established" fix nearby; re-confirm if separate rhetorical-Q opener remains.

## vol2/collective_communication/collective_communication.qmd

Applied: AlltoAll → AllToAll canonicalization (8 instances), traveler analogy at line 1641 replaced with technical description (kg unit issue moot), "an 100$\times$" → "a 100$\times$" article fix at line 1647, "sanctuary"/"bridge" colloquial metaphor at line 1242 replaced with direct bandwidth comparison, `DistributedDataParallel\index{}` markup leak inside backticks moved outside.

Deferred to discussion:

- **BF16 70B AllReduce dominates `L_lat` claim (line ~59)**: A 140 GB/50 GB/s transfer is bandwidth-bound, not latency-bound. **Action**: change to "dominates the bandwidth/data-movement term" or split contributions.
- **BF16 vs FP32 precision drift for 70B running example (lines 59/69 vs 157/511 vs 592)**: Chapter alternates precision. **Action**: declare each switch or pick one canonical precision.
- **Raw `\ref{pri-alpha-beta-model}` (line ~602)**: Use Quarto's `@pri-alpha-beta-model` cross-reference.
- **NCCL reality-gap table uncited (line ~602)**: Add source or relabel as illustrative model.
- **"can an algorithm achieve the best of both?" rhetorical Q (line ~950)**: Convert to declarative transition.
- **Fallacies bold paragraph starters (line ~2031)**: Operations-manual anti-pattern.
- **New compression-payback notebook after Summary opens (line ~2073)**: Move into compression trade-offs section.
- **"roads are constantly crumbling" colloquial chapter-connection (line ~2159)**: Replace with direct transition.
- **3.5x vs 5.7x compression-payback comment drift (line ~2082)**: Update source comment to match calculated value.

## vol2/distributed_training/distributed_training.qmd

Applied: Purpose body second-person → impersonal + "But" → "however" recast (line 28), Jeff Dean test callout second-person → impersonal (line 191), `nvlink_a100_str`-`nvlink_h100_str` hyphen → en-dash (`--`), GPT-2 notebook second-person + rhetorical Q → declarative (line 963 region), "When does parallelism hurt?" heading → "Critical batch size and diminishing returns" + opener recast (1307), pipeline-bubble Problem callout second-person → impersonal (1763 region).

Deferred to discussion:

- **"Exponentially" with cluster size (line ~160)**: Conflates expected failure rate (~linear) with system survival probability (~exponential). **Action**: specify which quantity, align scaling language.
- **Loop-transform diagram in `{text}` block (line ~223)**: `fig-loop-transforms` label and caption inside a text fence won't render as a numbered figure. **Action**: use TikZ/Mermaid/Graphviz or convert to SVG with a real `#fig-` div.
- **Slow-AllReduce debugging example arithmetic (line ~644)**: Numbers don't add up. 100ms observed vs 50ms predicted vs 120ms naive vs 240ms ring-AllReduce; "better than naive" + 40ms remaining gap inconsistent. **Action**: pick one baseline.
- **Footnote `-- ` ASCII double-hyphens as parenthetical breaks (line ~261)**: Should be em-dash or recast.
- **3D parallelism summary figures reuse same asset (line ~2618)**: Two `fig-` divs use `images/svg/3d-parallelism.svg` with different captions, one claiming sliced/hierarchical view. **Action**: distinct asset for `fig-3d-parallelism-sliced` OR revise caption.
- **Manual bold caption outside `fig-cap` (line ~2621)**: Duplicates Quarto's caption mechanism.

## vol2/data_storage/data_storage.qmd

Applied: second-person → impersonal in four `**Problem**:` / `**Scenario**:` / `**Systems conclusion**:` callout openers (lines 202, 651, 699, 1033, 1385 originals), `post-training` → `posttraining` in retention prose.

Deferred to discussion (all math/structural):

- **`@fig-access-patterns-vol2` figure-vs-caption mismatch (line ~417)**: Prose discusses sequential-vs-random access; figure caption describes I/O throughput vs model size. **Action**: align figure asset/caption with the prose, or revise prose to match the displayed figure.
- **HBM↔object-storage 300,000x bandwidth gap (line ~442)**: Adjacent table values imply ~33.5x, not 300,000x. Claim recurs in summary + takeaways. **Action**: recompute and standardize.
- **Storage pyramid caption vs `@tbl-storage-hierarchy-merged` mismatch (line ~448)**: Several tier values disagree. **Action**: pick one authoritative set.
- **Archive cost ratio 5x vs 20x (line ~912)**: Stated prices yield 5x, prose says 20x. **Action**: reconcile.
- **Two-year archive cost off by 10x (line ~918)**: Computed total should be \$9,600, prose says \$960. **Action**: correct dollar amount or volume.
- **ZeRO-3 70B per-node shard math (line ~1572)**: 140 GB per-node for FP16 weights of a 70B model across 1,000 nodes is inconsistent with sharding. **Action**: clarify checkpoint contents + sharding assumption, recompute.
- **175B per-node checkpoint 4 GB vs 6.8 GB (line ~1580)**: Multiple downstream calculations build from 4 GB but the implied math gives 6.8 GB. **Action**: pick one shard-size model.
- **Idle-accelerator cost \$9,600/day vs \$13,000/day (line ~1474)**: Callout disagrees with immediately preceding calculation. **Action**: align.
- **Retrieval-infrastructure subsection interrupts Fallacies/Pitfalls (line ~1647)**: Reorganize or convert to fallacy/pitfall.

## vol2/network_fabrics/network_fabrics.qmd

Applied: trailing colon-only QMD garbage stripped (lines 1662–1664), `us` → `μs` for microseconds in fig-cap/fig-alt at line 478, `Meta grand teton` → `Meta Grand Teton` heading at line 1360, `datacenter` → `data center` / `data-center` partial sweep.

Deferred to discussion:

- **Five-level taxonomy mismatch (line ~113)**: Figure caption's Level 2–5 labels don't match the section heading taxonomy in lines 373–377. **Action needed**: unify taxonomy.
- **Fat-tree scaling claims conflict (line ~859)**: Multiple host/switch counts cannot all be true under one topology. **Action needed**: recompute and distinguish two-tier leaf-spine from k-ary three-tier Clos.
- **Oversubscription dollar discrepancy (line ~963 vs ~1169)**: Hardcoded \$75M waste estimate contradicts later `BisectionBottleneck` ~\$142M. **Action needed**: align models.
- **Bandwidth-staircase figure mismatch (line ~1350)**: `@fig-hierarchical-staircase` rendered asset doesn't match the prose intent. **Action needed**: replace asset OR revise prose+cross-reference.
- **Rail-optimization data-vs-tensor parallelism confusion (line ~998)**: Mixes traffic patterns. **Action needed**: separate same-rank data-parallel AllReduce from tensor-parallel activation exchange.
- **Checkpoint over-explained (line ~1530)**: Diagnostic reasoning included in the prompt undermines exercise value. **Action**: shorten prompt or move reasoning to answer key.
- **Fallacies bold paragraph starters (line ~1542)**: Operations-manual anti-pattern. **Action**: convert to proper list/callout/prose topic sentences.

## vol2/compute_infrastructure/compute_infrastructure.qmd

Applied: heading hierarchy fix (`### Accelerator spectrum` → `## Accelerator Spectrum`), four rhetorical-question section transitions converted to declarative (lines 187/197/213/227 originals), arithmetic-intensity callout opener rewritten with full definition sentence, two additional rhetorical questions in Roofline + Benchmarking sections (855/1459) converted, hyphen→en-dash ranges, `50g`/`400g` → `50 g`/`400 g`, `datacenter` → `data center` / `data-center` partial sweep, Summary fragment joined, "But"→"however" recast in chapter-connection.

Deferred to discussion:

- **Roofline ridge-point inconsistency (line ~829 region)**: Footnote says ~295 FLOP/byte (FP16 H100); later figure caption and prose use ~591 FLOP/byte. The two values reflect different precisions (FP16 vs FP8 peak). **Action needed**: choose one precision for the H100 Roofline example and make footnote, figure caption, plotted value, and prose agree.
- **Cooling rack-count inconsistency (line ~2231 region)**: Cooling paragraph says "128-rack training cluster"; nearby rack arithmetic uses 1,024 GPUs as 32 racks. **Action needed**: reconcile to one example scale.
- **TCO figure-caption inconsistency (line ~2811)**: Figure caption describes 1,000-node H100 cluster (8,000 GPUs); surrounding prose analyzes a 1,000-GPU/125-node cluster; caption's cloud cost matches the smaller scale. **Action needed**: align caption, plotted data, and prose totals.
- **Citation gaps in environmental-impact paragraph (line ~2165)**: GPT-3 energy estimate, carbon-intensity values, Microsoft nuclear procurement, Google geothermal investment are uncited.
- **Bold step starters in planning-methodology example (line ~3073)**: `**Step 1**`, `**Step 2**`, … inside a long body paragraph. Flagged as operations-manual style. **Options**: (a) convert to flowing prose; (b) restructure as proper numbered list with full sentences.

## vol2/introduction/introduction.qmd (major structural items deferred)

This is the largest single deliverable in vol2. YAML reports 11 findings; mechanical fixes applied (Latent-Bound → latency-bound at line 816, "datacenters" → "data centers" co-located). The structural items below need user input:

- **Foundational Concepts section reintroduces frameworks (line 1675)**: C³, Fleet Stack, roadmap, AI Triad, Five-Pillar Framework, systems principles, and archetypes are all re-presented after their initial development earlier in the chapter. **Options**: (a) consolidate the repeated framework overview into earlier sections; (b) make line 1675 a brief synthesis-and-forward-pointer rather than a re-presentation; (c) keep current structure if a final-section synthesis is pedagogically desired.
- **C³ taxonomy figure duplicated (line 1679)**: One TikZ at `@fig-c3-taxonomy`, one SVG at `@fig-vol2-c3-taxonomy`. **Action needed**: keep one canonical, update cross-references in lockstep.
- **Lighthouse archetypes introduced twice**: Once in a callout near line 328, again at the canonical section + table at lines 1910–1934. **Options**: (a) replace the early callout with a forward pointer to `@sec-vol2-introduction-archetypes`; (b) move the canonical table near the first mention.
- **Cloud Archetype vs Archetype A naming drift (line 806)**: Line 806 says Frontier Training "inherits the Cloud Archetype" and uses GPT-4. The canonical roster (line 1912+) defines Archetype A as GPT-4/Llama-3 and does not include a Cloud Archetype. **Action needed**: align line 806 with the canonical Archetype A label, or define Cloud Archetype as a separate taxonomy.
- **C³ corner mapping inconsistency (line 1912)**: Archetype intro says each archetype stresses Communication, Coordination, or Compute, but the table maps the archetypes via Throughput Bound / Volume & Latency / Power & Privacy. No archetype clearly occupies Compute. **Action needed**: choose one constraint vocabulary for the canonical table.
- **Scale-Moment magnitude inconsistency (line 227 vs line 103)**: Table caption says seven orders of magnitude; line 103 references 10²⁵–10²⁶ FLOPs for frontier (suggesting eight orders from 10¹⁸). **Action needed**: reconcile.
- **Purpose paragraph density (line 28)**: Long opening paragraph stacks emphatic italicized When/What constructions. **Options**: (a) split into two paragraphs and reduce italic density; (b) keep as-is — the manifesto register may be intentional for a volume opener.
- **Fleet Stack + Roadmap figures duplicate four-part progression (line 1723)**: Adjacent figures present the same vertical structure. **Options**: (a) merge into one figure; (b) keep both if they serve distinct pedagogical purposes.
- **Six Systems Engineering Principles bullet-list format (line 1901)**: Bold-label paragraphs flagged as operations-manual style. **Options**: (a) keep as-is for compact reference; (b) move to a callout; (c) replace with synthesis paragraph.
