# Float exposition eval — data_engineering.qmd (vol1)
Standard: FLOAT_EXPOSITION_STANDARD.md (caption excluded from prose budget)

## Summary
| type | level | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|---|
| equation | 🔴 | 8 | 6 | 2 | 0 |
| algorithm | 🔴 | 0 | — | — | — |
| table | 🟠 | 10 | 6 | 4 | 0 |
| figure | 🟠 | 15 | 13 | 2 | 0 |
| listing | 🟡 | 4 | 1 | 3 | 0 |
| **total** | | **37** | **26** | **11** | **0** |

---

## Findings (⚠️ only — no 🛑 found)

---

### ⚠️ `eq-label-entropy` (equation 🔴) — def L2486

- **Ref (body prose):** "For systems with calibrated model probabilities, label confidence entropy provides an alternative detection signal. Let $p_i$ represent the model's probability assigned to label category $i$. @Eq-label-entropy defines this measure:"
- **Payoff (body prose):** "Rising entropy in model confidence distributions suggests increasing ambiguity or mislabeling in training data, as the model learns from inconsistent supervision."
- **Missing moves:** (1) The measure $H_{\text{label}}$ is never named in words — the prose gives the variable name but never says what value range is actionable. (2) No regime implication: the standard requires stating a "consequence or regime it implies" — e.g., at what entropy level does ambiguity become a training concern? The single payoff sentence states a direction ("rising suggests") but gives no threshold or reference point to make it actionable. Compare with `eq-cohens-kappa`, which names a concrete threshold ($\kappa < 0.4$) immediately after.
- **Suggested rewrite (no em-dash/hyphen, at most one colon per para, content leads):**
  ```diff
  - Rising entropy in model confidence distributions suggests increasing ambiguity or mislabeling in training data, as the model learns from inconsistent supervision.
  + Label confidence entropy measures how spread the model's probability mass is across classes: when entropy is low the model assigns confident labels, and when entropy rises toward its maximum of $\log C$ for $C$ classes the model is uncertain across nearly all categories. A sustained upward trend in $H_{\text{label}}$ over successive training batches signals that the model is receiving increasingly inconsistent supervision, typically because annotators are mislabeling ambiguous examples or because the boundary between classes has shifted in the underlying data.
  ```

---

### ⚠️ `eq-data-supply` (equation 🔴) — def L4166

- **Ref (body prose):** "The bottleneck relation in @eq-training-throughput captures this limit, with the data supply rate defined in @eq-data-supply. The min-of-rates form is the $T_{\text{step}} = \max(T_{\text{compute}}, T_{\text{io}})$ inequality from @sec-data-engineering-data-ingestion-8efc in different notation: the stage with the larger time is the stage with the smaller rate, and either way it sets the pace."
- **Missing move:** The `Overhead` term in `Data Supply Rate = Storage Bandwidth × (1 − Overhead)` is never defined or named in body prose. The standard requires that every symbol be named (a "where" clause is fine). The payoff paragraph correctly addresses the compound result but never explains what `Overhead` represents. A reader who reads only the prose cannot know whether overhead refers to decompression cost, metadata reads, format parsing, or OS buffering.
- **Suggested rewrite (adding the missing where-clause inline):**
  ```diff
  - The bottleneck relation in @eq-training-throughput captures this limit, with the data supply rate defined in @eq-data-supply. The min-of-rates form is the $T_{\text{step}} = \max(T_{\text{compute}}, T_{\text{io}})$ inequality from @sec-data-engineering-data-ingestion-8efc in different notation: the stage with the larger time is the stage with the smaller rate, and either way it sets the pace.
  + The bottleneck relation in @eq-training-throughput captures this limit, with the data supply rate defined in @eq-data-supply, where Overhead represents the fraction of storage bandwidth consumed by format parsing, decompression, metadata reads, and OS buffering rather than useful feature bytes. The min-of-rates form is the $T_{\text{step}} = \max(T_{\text{compute}}, T_{\text{io}})$ inequality from @sec-data-engineering-data-ingestion-8efc in different notation: the stage with the larger time is the stage with the smaller rate, and either way it sets the pace.
  ```

---

### ⚠️ `fig-keywords` (figure 🟠) — def L818

- **Ref (body prose):** "As @fig-keywords illustrates, a KWS system operates as a lightweight, always-on front-end that triggers more complex voice processing systems. Even this seemingly simple architecture surfaces interconnected challenges across all four pillars: Quality (accuracy across diverse environments), Reliability (consistent battery-powered operation), Scalability (severe memory constraints), and Governance (privacy protection). These constraints limit KWS systems to a few dozen languages: collecting high-quality, representative voice data for smaller linguistic populations proves prohibitively difficult. All four pillars must work together to achieve successful deployment."
- **Payoff (body prose):** "The four pillars translate directly into engineering constraints for the KWS system."
- **Missing move:** The lead-out is a pivot sentence that restates the section topic rather than interpreting what the figure *demonstrates*. The figure shows a user interacting with an Echo Dot through a wake-word exchange, illustrating the two-stage architecture (always-on detector triggers full assistant). Neither the citation sentence nor the payoff sentence tells the reader what to conclude from that visual: that the always-on stage must run at microcontroller power budgets while the triggered stage can use full cloud compute, which is why the data engineering constraints differ so sharply between the two. The "so what" of the figure lives only in the four-pillar list, without stating the architectural implication the figure is meant to make concrete.
- **Suggested rewrite (lead-out added after the float, replacing the pivot sentence):**
  ```diff
  - The four pillars translate directly into engineering constraints for the KWS system.
  + The figure makes the two-stage architecture concrete: the always-on detector must run continuously on a microcontroller's milliwatt power budget, while the triggered assistant has access to cloud-scale compute. That power asymmetry is why data engineering for KWS imposes far stricter constraints at the first stage than at the second — the model, its features, and its runtime footprint must all fit within a fixed memory and energy envelope that leaves no room for the data movement patterns acceptable at cloud scale.
  ```

---

### ⚠️ `fig-pipeline-flow` (figure 🟠) — def L2138

- **Ref (body prose):** "@Fig-pipeline-flow maps that end-to-end path across data sources, ingestion, processing, labeling, storage, and ML training."
- **Payoff (body prose):** "Each layer plays a specific role in the data preparation workflow. Selecting appropriate technologies requires understanding how our four framework pillars manifest at each stage. Quality requirements at one stage affect scalability constraints at another, reliability needs shape governance implementations, and the pillars interact to determine overall system effectiveness."
- **Missing move:** The payoff states that layers interact and pillars apply, but never delivers the mechanism the figure demonstrates: that each layer scales independently, which is the architectural property that enables modular quality control. The key result of the figure — that the governance band spans the whole pipeline while the functional layers can be replaced or scaled independently — is never stated in prose. The reader is told that "pillars interact to determine overall system effectiveness," which could describe any system diagram.
- **Suggested rewrite:**
  ```diff
  - Each layer plays a specific role in the data preparation workflow. Selecting appropriate technologies requires understanding how our four framework pillars manifest at each stage. Quality requirements at one stage affect scalability constraints at another, reliability needs shape governance implementations, and the pillars interact to determine overall system effectiveness.
  + Because each layer in the pipeline scales independently, a team can swap the ingestion tier (adding a streaming path for real-time audio) without touching the storage or processing tiers, and can strengthen governance controls across the full pipeline without redesigning any functional layer. That modular independence is the architectural property that makes large-scale data engineering tractable: quality enforcement, schema validation, and access controls can be applied at each layer boundary rather than forcing a single monolithic transformation that would couple all concerns together.
  ```

---

### ⚠️ `lst-data-expectations` (listing 🟡) — def L2354

- **Ref (body prose):** "teams codify quality expectations as executable assertions (@lst-data-expectations) that run on every pipeline execution."
- **Payoff (body prose):** "Once validation becomes code, expectation suites become versioned artifacts alongside training code. When training code changes, expectation updates keep data contracts evolving with it. This coupling reduces the risk of silent divergence where code assumes data properties that the upstream pipeline no longer provides."
- **Missing move:** The citation names the mechanism abstractly ("executable assertions") but the standard for listings requires that prose frame *what the code shows* — the mechanism it embodies and what the reader should notice. The key design choice in the listing is that column-level type checks, range checks, and missing-value checks are expressed as named expectations (not ad-hoc conditionals), so they can be tracked, versioned, and reported together. Neither the citation sentence nor the payoff paragraph names that design choice; a reader who skips the code learns only that "assertions run on every pipeline execution."
- **Suggested rewrite (adding orientation before the listing):**
  ```diff
  - teams codify quality expectations as executable assertions (@lst-data-expectations) that run on every pipeline execution.
  + teams codify quality expectations as executable assertions that run on every pipeline execution (@lst-data-expectations). Notice how each expectation is a named, self-describing contract — `expect_column_values_to_not_be_null`, `expect_column_values_to_be_between` — rather than a bare conditional. That naming convention is the design choice that matters: named expectations compose into a versioned expectation suite, so schema changes and contract violations surface as structured diffs rather than uncaught exceptions.
  ```

---

### ⚠️ `lst-etl-elt-cost-comparison` (listing 🟡) — def L3062

- **Ref (body prose):** "Deciding where to run these transformations involves a small cost model for the cost of transformation placement, as shown in @lst-etl-elt-cost-comparison."
- **Payoff (body prose):** Pivots to CAP theorem and streaming systems, never returning to the listing.
- **Missing move:** The citation is a bare pointer ("as shown in"). No prose names what the code *demonstrates* — the key finding is that ETL's upfront compute cost dominates at high schema-change frequency, while ELT's higher storage cost dominates at high feature-iteration rate. Neither the citation sentence nor any post-listing paragraph delivers that conclusion; the reader must infer it from the code itself.
- **Suggested rewrite (citation sentence replaced; a lead-out sentence added after the listing):**
  ```diff
  - Deciding where to run these transformations involves a small cost model for the cost of transformation placement, as shown in @lst-etl-elt-cost-comparison.
  + Deciding where to run these transformations requires comparing the recurring compute cost of ETL's pre-load transformations against the higher storage cost ELT incurs by retaining raw data (@lst-etl-elt-cost-comparison). The cost model shows that ETL is cheaper when schemas are stable and transformation logic changes rarely, because the upfront compute is amortized across many training runs. ELT breaks even or wins when feature definitions evolve frequently, because reprocessing from raw SQL is faster than replaying a distributed pipeline transformation.
  ```

---

### ⚠️ `lst-dvc-workflow` (listing 🟡) — def L4385

- **Ref (body prose):** "@Lst-dvc-workflow shows how DVC provides Git-like semantics for data versioning, while @lst-delta-time-travel demonstrates querying historical data states directly in SQL." (This citation appears *after* the listing at L4403, not before it.)
- **Payoff (body prose — appears before the citation, at L4401):** "Data versioning is the storage analogue of source-control provenance. It connects model versions to exact training data, enabling debugging and reproducibility. Without it, teams cannot identify the exact data that trained the model now misbehaving in production."
- **Missing moves:** (1) The listing lacks a forward citation — it appears before any prose references it. The standard requires that floats be introduced before they appear. (2) The citation at L4403, which follows the listing, names the mechanism ("Git-like semantics") but does not name what to *notice* in the code: the pairing of `git checkout` with `dvc checkout` is the key design detail that makes the mechanism concrete, and the fact that DVC stores a `.dvc` metadata file in Git rather than the data itself is the insight that makes large-file versioning tractable. Neither observation is in prose.
- **Suggested rewrite (adding a lead-in before the listing and improving the post-listing prose):**
  ```diff
  - [no prose before lst-dvc-workflow — float appears first]
  + The key design choice in DVC is to version a small metadata pointer file (the `.dvc` file) inside Git while the actual data bytes live in a separately configured remote store. @Lst-dvc-workflow shows this pairing in action: `git checkout abc123` restores the pointer, and `dvc checkout` then fetches the corresponding data, so any historical dataset state can be recovered with the same two-command discipline that software teams use for code.
  ```
  Replace the post-listing sentence at L4403:
  ```diff
  - @Lst-dvc-workflow shows how DVC provides Git-like semantics for data versioning, while @lst-delta-time-travel demonstrates querying historical data states directly in SQL.
  + @lst-delta-time-travel extends the same point to structured data: Delta Lake's time-travel syntax lets a SQL query address any historical snapshot by date or version number, achieving point-in-time correctness without a separate snapshot management layer.
  ```

---

### ⚠️ `tbl-kws-design-space` (table 🟠) — def L1054

- **Ref (body prose):** "@Tbl-kws-design-space quantifies key trade-offs, enabling principled decisions rather than ad-hoc selection. One row uses mel-frequency cepstral coefficients (MFCCs)… compact speech-frequency features whose coefficient count controls feature size, compute cost, and acoustic detail."
- **Payoff (body prose):** "A concrete budget scenario shows how to apply this design space analysis."
- **Missing move:** The payoff is a one-sentence bridge with no content drawn from the table. The standard requires that prose name the load-bearing contrast or the specific row(s) that matter. The dominant result in this table — that the memory constraint eliminates cloud inference outright (64 KB limit vs. unlimited for cloud) and that synthetic augmentation is ten times cheaper than equivalent real data collection — is never stated in body prose. Both facts reside only in the cells and caption.
- **Suggested rewrite (replacing the payoff bridge sentence):**
  ```diff
  - A concrete budget scenario shows how to apply this design space analysis.
  + Two rows in the table drive the most consequential decisions. Memory constraint (64 KB) eliminates cloud inference regardless of accuracy preference: the network stack alone exceeds the budget, so local inference is not a trade-off but a requirement. Synthetic augmentation, at ten times the cost efficiency of equivalent real-data collection, makes data diversity achievable within the labeling budget — the table's central engineering insight is that the binding constraint is memory, and the dominant lever within that constraint is the synthetic-to-real data ratio.
  ```

---

### ⚠️ `tbl-data-engineering-cost-constants` (table 🟠) — def L1605

- **Ref (body prose):** "ML engineers should internalize the data engineering constants in @tbl-data-engineering-cost-constants and @tbl-data-engineering-time-constants. The pattern that emerges is that labeling consistently dominates storage and compute costs, so teams should reason from cost ratios rather than isolated prices."
- **Payoff (body prose — immediately after the table):** "@Tbl-data-engineering-time-constants extends the picture with characteristic durations for labeling, training, and serving operations."
- **Missing move:** The post-table sentence pivots to the next table without delivering a result from this one. The citation sentence identifies the pattern ("labeling dominates") but names no specific rows or ratios. The H&P standard requires a "the key result is…" sentence. From the table: expert medical labeling at hundreds of dollars per study is roughly four to five orders of magnitude more expensive than S3 storage per TB-month — the ratio that makes the dominance claim concrete. That quantified contrast lives only in the cells.
- **Suggested rewrite (replacing the pivot sentence after the table):**
  ```diff
  - @Tbl-data-engineering-time-constants extends the picture with characteristic durations for labeling, training, and serving operations.
  + The table makes the dominance concrete: expert medical labeling exceeds standard cloud storage cost by roughly four orders of magnitude, and even crowdsourced labeling costs three to four times more per sample than the storage needed to retain that sample for a year. A team that adds one thousand labeled medical images adds a cost that could store the entire resulting dataset for years. This asymmetry is why labeling pipeline efficiency is a higher-leverage investment than storage optimization for most ML projects. @Tbl-data-engineering-time-constants extends the picture with characteristic durations across the pipeline lifecycle.
  ```

---

### ⚠️ `tbl-storage-performance` (table 🟠) — def L3915

- **Ref (body prose):** "@Tbl-storage-performance reveals why ML systems employ tiered storage architectures. Consider the economics of storing our KWS training dataset ([dataset_size]): object storage costs [cost], enabling affordable long-term retention of raw audio, while maintaining working datasets on NVMe for active training costs [cost range] but provides [load_speedup] faster data loading."
- **Payoff (body prose):** `[^fn-nvme-gpu-utilization]` footnote definition (not body prose). No body paragraph follows before the next code cell.
- **Missing move:** The citation delivers a specific KWS scenario but names only one trade-off (object storage vs. NVMe). No prose draws the general conclusion from the table — that each tier maps to a different ML workload stage, and that using the wrong tier creates order-of-magnitude penalties that cannot be recovered through software. That conclusion lives only in the caption. After the table closes, there is no lead-out paragraph in body prose at all.
- **Suggested rewrite (adding a lead-out paragraph after the table definition):**
  ```diff
  - [no body prose after tbl-storage-performance; next block is a footnote then a python cell]
  + The table encodes a matching rule: training data loading requires high sequential throughput (NVMe at 5–7 GB/s), feature serving requires low random-read latency (in-memory cache at 1–10 μs), and archival compliance storage requires low cost per TB (Glacier at $1–4). Placing training data on object storage reduces cost by a factor of four to fifteen but cuts sequential throughput by ten times or more, an order-of-magnitude performance penalty that no prefetching strategy fully recovers. Tiered storage exists because no single tier can satisfy all three requirements simultaneously.
  ```

---

### ⚠️ `tbl-data-debt-metrics` (table 🟠) — def L4490

- **Ref (body prose):** "@Tbl-data-debt-metrics provides quantitative warning and critical thresholds for each debt category, making data debt trackable as an engineering SLO."
- **Payoff (body prose):** "Data debt compounds through feedback loops unique to ML systems:" (followed by a bullet list describing the four debt categories, not the table's threshold values).
- **Missing move:** The citation is a float-announcer: it names what the table holds but draws no conclusion from its values. The lead-out (L4492) recaps debt categories already introduced in the preceding four paragraphs rather than stating the key result the table encodes — specifically, that the PSI threshold of 0.25 is the single most actionable early-warning indicator because it correlates with measurable accuracy loss within one to two retraining cycles, giving teams a numeric trigger rather than a judgment call. That insight lives only in the caption.
- **Suggested rewrite (replacing the announcer citation):**
  ```diff
  - Unlike technical debt, which can be assessed through code complexity metrics, data debt requires specialized measurement approaches. @Tbl-data-debt-metrics provides quantitative warning and critical thresholds for each debt category, making data debt trackable as an engineering SLO.
  + Unlike technical debt, which can be assessed through code complexity metrics, data debt requires specialized measurement approaches. @Tbl-data-debt-metrics converts each debt category into a trackable engineering SLO with explicit warning and critical thresholds. The most operationally useful threshold is PSI above 0.25 against the training baseline: at that level, distribution divergence typically correlates with measurable accuracy loss within one to two retraining cycles, giving teams a numeric trigger for remediation rather than a judgment call about whether drift is "bad enough."
  ```

---

## Dangling refs (scanner-reported, not graded as floats)
The scanner reports three dangling refs — floats cited but not defined in this file:
- `@fig-ds-time` (L87) — defined in another chapter
- `@tbl-dam-taxonomy` (L87) — defined in another chapter
- `@eq-degradation` (L2418) — defined in another chapter

These are cross-chapter references, not missing-exposition findings.
