# Synthesized Candidate Method for the ML Systems Taxonomy

This specification synthesizes the top-ranked elements from the blind three-family method study (`Method-A` by `gpt-6-astra`, `Method-B` by `claude-opus-5-5`, and audited components of `Method-C` by `gemini-4-argon-high`) into a single operational methodology for constructing, validating, and maintaining the ML Systems Taxonomy.

Every adopted mechanism below cites the blind proposal from which it was drawn, the rubric dimension (`D1`–`D8` in `taxonomy/method-study/rubric.yaml`) on which that proposal scored highest, and whether the threshold or rule is derived from a published standard or a project-calibrated parameter (`[P]`).

---

## 1. End-to-End Construction Pipeline

**Source provenance**: Synthesized from `Method-B` §1 (14-step DAG with explicit input/output artifacts, stopping rules, and failure recovery, scored 5.0/5.0 on `D1`) and `Method-A` §1 (8-stage pipeline with split discipline and pre-registration gates, scored 4.5–5.0/5.0 on `D1`).

| Stage | Name | Inputs | Outputs | Stopping Condition & Recovery |
| :--- | :--- | :--- | :--- | :--- |
| **S0** | Charter, Scope & Pre-registration | Domain boundary, downstream application requirements, frozen rubric | `charter.yaml`, `preregistration.yaml` | Scope exclusion tests defined; evaluation splits frozen before candidate harvesting begins. |
| **S1** | Stratified Corpus Frame & Split Allocation | Enumerated source frame across 5 strata (`S1`–`S5`) | `corpus_manifest.json` with `dev`, `cal`, `test`, and `reserve` cluster assignments | Every stratum represented; provenance-family clusters (`cluster_id`) isolated across splits (`Method-A` §3.3). |
| **S2** | Multi-Extractor Candidate Harvesting | `dev` split documents only | `candidate_mentions.jsonl` with exact source locators and verbatim evidence spans | Extractors run in isolated sessions without access to `cal` or `test` splits or existing vocabulary concepts (`Method-A` §1, `Method-B` §1 Step 3). |
| **S3** | Lexical Normalization & Cluster Deduplication | `candidate_mentions.jsonl` | `candidate_clusters.json` | Z39.19 Clauses 6–7 normalization (singular count nouns vs. mass nouns, parenthetical qualifiers for homographs); lexical/synonym grouping. |
| **S4** | Warrant Verification & Provenance Independence | `candidate_clusters.json`, `corpus_manifest.json` | `warranted_candidates.json`, `unwarranted_backlog.json` | Candidate passes the two-source provenance-family independence gate (§2.1); single-source items route to backlog. |
| **S5** | Concept-vs-Alias & Post-Coordination Adjudication | `warranted_candidates.json` | `admitted_concepts.yaml`, `uf_aliases.yaml`, `split_recipes.yaml` | Each candidate evaluated against the 4-part Concept-vs-Alias and Compound Splitting Protocol (§2.3–2.4). |
| **S6** | Primary Backbone Placement & Typed Relations | `admitted_concepts.yaml` | `backbone_tree.yaml`, `relations.yaml` | Every concept receives one home subject (`home_subject_id`), optional `specializes` (`BTG`/`BTP`/`BTI` tested), and constrained associative relations with justification templates (`Method-A` §4.1–4.3, `Method-B` §4.1–4.3). |
| **S7** | Scope Note Authoring (Genus-Differentia + Boundary) | `admitted_concepts.yaml`, `relations.yaml` | Concept records with `description` / `scope_note` | Every scope note states the immediate genus, distinguishing system mechanism, and explicit boundary exclusions (`Method-B` §1 Step 9). |
| **S8** | Automated Structural & Schema Gate | Concept, subject, relation, and overlay files | `check_report.json` | Zero errors across all compiled structural invariants (§4.4). |
| **S9** | Double-Judged Blind Coverage & Indexing Calibration | Frozen vocabulary, `cal` / `test` split | `coverage_report.json`, `agreement_report.json` | Two-pass blind coverage + seeded-gap sensitivity check (§5.1) and multi-rater agreement ($\kappa$, $\alpha_{\text{MASI}}$, Hooper, Rolling, $hF_1$) coupled with specificity counter-metrics (§5.2). |

---

## 2. Concept Admission, Warrant, and Boundary Rules

**Source provenance**: Synthesized from `Method-A` §2 (`D2` = 4.5–5.0) and `Method-B` §2 (`D2` = 4.5–5.0).

### 2.1 Multi-Type Warrant with Provenance-Family Independence
Following ANSI/NISO Z39.19-2005 Clause 5.3.5 and ISO 25964-1:2011 Clause 6, admission requires explicit warrant:
1. **Two-Source Provenance-Family Rule**: A regular concept requires documented literary warrant in **at least two independent external sources** (`provenance_family_id` must differ). Two sources share a `provenance_family_id` (and count as a single source) if:
   - Their author sets overlap by $\ge 50\%$ (`Method-A` §2.1, `[P]`),
   - One document is a textbook, lecture slide deck, or blog adaptation authored by the creators of the other, or
   - One is a benchmark specification and the other is the paper introducing that benchmark (`Method-B` §2.2).
2. **External vs. Project-Internal Warrant**: Project-internal application corpora (such as internal question banks or assessment items) provide **organizational/user warrant leads**, not independent external literary warrant. A term appearing only in an internal question bank + one external source must either secure a second independent external source or be explicitly marked as a single-source curriculum-anchored exception (`Method-B` §2.1).
3. **Foundational Single-Source Exception**: Permitted only when a concept is the primary subject of a dedicated section in the governing textbook or an authoritative national/international standard (e.g., NIST AI 600-1, ISO/IEC 42001, IEEE 7000), recorded with an explicit `warrant_exception_reason`.

### 2.2 Named Entity & Product Exclusion Gate (Z39.19 Clause 6.3.3)
Named commercial products, specific software packages, hardware SKUs, model releases, and benchmark suites (`vLLM`, `TensorRT-LLM`, `DeepSpeed`, `NVIDIA H100`, `Llama-3`) are **excluded as general taxonomy concepts** (`Method-A` §2.2, `Method-B` §2.4).
- **Abstraction rule**: Extract the underlying architectural pattern, algorithm, or system mechanism (e.g., `vLLM` $\rightarrow$ *paged attention* and *continuous batching*; `ZeRO` $\rightarrow$ *sharded optimizer state partitioning*).
- **Alias rule**: Do not map a product name as an exact synonym (`alt_label`) of a general mechanism if the product implements multiple mechanisms (`Method-B` §2.4).
- **Instance exception**: A named standard or protocol is admissible only when it functions as an open architectural interface standard across independent implementations (e.g., `OpenTelemetry` or `ONNX` when scoped to the open interchange specification, tested via `BTI`).

### 2.3 Operational 4-Part Concept-vs-Alias Decision Protocol
To replace subjective editorial judgment on whether two terms $X$ and $Y$ are synonyms (`alt_label` / `UF`) or distinct concepts (`Method-A` §2.3, `Method-B` §1 Step 10 Test A & §2.5), editors apply four sequential tests:
1. **Extension & Failure-Mode Identity Test**: Do $X$ and $Y$ denote the same set of system mechanisms, architectural components, or failure modes across the target corpus?
2. **Differential Indexing Test**: Can the proposer cite at least one concrete syllabus unit, paper section, or diagnostic scenario where indexing with $X$ is informative and indexing with $Y$ would be inaccurate or misleading?
3. **Operational Distinction Test**: Do $X$ and $Y$ differ in at least one conserved system property (e.g., online vs. offline execution, hardware layer, asymptotic memory/compute trade-off, or control-plane vs. data-plane placement)?
4. **Warrant Floor on Both Sides**: Do both $X$ and $Y$ independently satisfy the two-source warrant rule?

**Decision rule**: $X$ and $Y$ are admitted as **separate concepts** if and only if Tests 2, 3, and 4 all hold. Otherwise, the lower-frequency or less standard surface form is recorded as an `alt_label` (`UF`) on the preferred concept.

### 2.4 Compound Term Splitting vs. Pre-Coordination (Z39.19 Clause 7, ISO 25964-1 Clause 7.3)
A multi-word compound term $C$ (e.g., *quantized KV-cache offloading*) is retained as a single pre-coordinated concept **only if** at least one of the following holds (`Method-A` §2.3, `Method-B` §2.5):
1. **Non-compositional semantics**: The meaning of $C$ is not the intersection of its constituent concepts.
2. **Lexicalized mechanism with distinct trade-offs**: $C$ names a specific algorithm, protocol, or system mechanism with its own failure modes and literature identity (e.g., *speculative decoding*, *continuous batching*, *activation checkpointing*).
3. **High co-occurrence utility**: $C$ appears as an atomic unit across $\ge 3$ independent provenance families (`[P]`) and splitting it would lose the specific engineering relationship between the modifiers.

Otherwise, $C$ is factored into existing atomic concepts (`USE A + B` post-coordination recipe). In particular, **metric $\times$ workload** combinations (e.g., *serving P99 latency*, *training throughput*) and **mechanism $\times$ generic modifier** combinations are factored rather than admitted as distinct concepts.

---

## 3. Enumerated Sampling Frame & Split Hygiene

**Source provenance**: Synthesized from `Method-A` §3 (`D3` = 5.0) and `Method-B` §3 (`D3` = 4.0–4.5). Full operational details are in `taxonomy/method-study/sampling-frame.md`.

1. **Five Stratified Source Populations**:
   - `S1` — Textbooks & Reference Monographs (curriculum and conceptual backbone warrant)
   - `S2` — Graduate & Advanced Undergraduate University Courses (pedagogical syllabus warrant)
   - `S3` — Peer-Reviewed Systems & ML Venues (MLSys, OSDI, SOSP, NSDI, ASPLOS, ISCA, NeurIPS/ICML/ICLR systems tracks; literary warrant)
   - `S4` — Practitioner Artifacts, Postmortems, Engineering Design Docs & Role Archetypes (operational and diagnostic warrant)
   - `S5` — Standards, Regulatory Frameworks & Benchmark Specifications (normative and assurance warrant: NIST, ISO/IEC, IEEE, OWASP, MLPerf)
2. **Cluster-Level Split Allocation**:
   - Every source unit is assigned a `provenance_family_id` (`cluster_id`) before random splitting.
   - Splits (`dev` for concept discovery, `held_out_test` for blind coverage and agreement evaluation, and `reserve` for replacing burned test slices) are drawn at the **cluster level** using a fixed PRNG seed recorded in `preregistration.yaml` (`Method-A` §3.3).
3. **Burned-Split Rule**:
   - Once a `held_out_test` slice is inspected to diagnose specific missing concepts and those concepts are added to the taxonomy, that slice is marked **burned** for measuring generalizable coverage of the resulting patch (`Method-A` §3.3, `Method-B` §3.4).

---

## 4. Structural Invariants, Relationships, and Lifecycle Governance

**Source provenance**: Synthesized from `Method-A` §4 (`D4` = 4.5–5.0) and `Method-B` §4 (`D4` = 4.5).

### 4.1 Polyhierarchy with Single Primary Backbone (`home_subject_id`)
- Every concept belongs to exactly one primary subject (`home_subject_id`) forming a strict spanning tree for navigation, rollups, and ownership (`Method-A` §4.1, `Method-B` §4.2).
- Cross-cutting subjects link existing concepts via `secondary_concepts` rather than minting duplicate concept IDs.
- Concept-to-concept `specializes` (`skos:broader`) forms a cycle-free DAG that must pass either the **generic (`BTG`) all-some is-a test** ("Every instance of $C$ is an instance of $P$ in all contexts") or the **partitive (`BTP`) inherent structural part test** (Z39.19 Clause 8.3).

### 4.2 Constrained Associative Relations (`related` / `RT`)
To prevent "everything-is-related-to-everything" graph drift (`Method-A` §4.3, `Method-B` §4.3):
- Every `prerequisite`, `contrasts_with`, and `related` edge must be disjoint from the transitive closure of `specializes` (`S27` / Z39.19 Clause 8.4).
- Every associative edge requires an explicit rationale sentence matching one of five typed templates:
  1. `producer_consumer` ($A$ produces artifact consumed by $B$)
  2. `tradeoff_dual` ($A$ and $B$ trade off on a shared resource bound)
  3. `mitigates_failure_of` ($A$ detects or mitigates a failure mode of $B$)
  4. `co_deployed_in_pipeline` ($A$ and $B$ are coupled stages in the same runtime pipeline)
  5. `evaluates_or_verifies` ($A$ measures or bounds property $B$)

### 4.3 Concept Scope Notes (`description`)
Every concept scope note follows a two-part **Genus–Differentia + Boundary** structure (`Method-B` §1 Step 9):
- **Clause 1 (Genus + Mechanism)**: States what kind of concept it is (algorithm, runtime mechanism, hardware structure, failure mode, assurance artifact, metric) and how it operates.
- **Clause 2 (Boundary / Contrast)**: Where potential overlap exists with a sibling or neighbor, states what the concept excludes or how it differs.

### 4.4 Automated Structural Gate (`binder concepts check`)
All existing structural invariants (`S14` preferred label uniqueness within scope, `alt_label` disjointness, `specializes` DAG acyclicity, home-rule disjointness, dangling reference zero-tolerance, single-child warning/justification) remain hard CI gates.

### 4.5 Lifecycle Operations & Typed Forward-Resolution Classes
Concept IDs (`mlsys.*`) are permanent and never reused (`Method-A` §4.6, `Method-B` §4.7–4.8):
- **Label refinement** (same meaning/extension): Update `label`, move old label to `alt_label`.
- **Meaning-changing scope shift**: If a proposed change alters which items are indexed by $>20\%$ (`[P]`), retire the old ID and mint a new ID rather than silently mutating historical tags.
- **Typed retirement resolution (`resolution_class`)**:
  - `auto` (1-to-1 exact replacement): Downstream tags can be migrated automatically to `replaced_by`.
  - `broaden` (N-to-1 merge into broader parent): Downstream tags migrate with a loss-of-specificity notice.
  - `review` (1-to-N split into narrower children): Downstream tags are flagged for human/automated re-indexing across the new children (`split_into`) rather than silently assigned to only one child.

---

## 5. Evaluation Protocol, Metrics, and Anti-Gaming Controls

**Source provenance**: Synthesized from `Method-A` §5–6 (`D5` = 5.0, `D6` = 4.5–5.0) and `Method-B` §5–6 (`D5` = 4.5, `D6` = 4.5–5.0).

### 5.1 Two-Pass Blind Coverage Protocol with Seeded-Gap Sensitivity
To eliminate single-judge confirmation bias when measuring coverage on held-out syllabi, papers, and standards (`Method-A` §5.6, `Method-B` §5.3):
- **Pass 1 (Vocabulary-Blind Topic Extraction)**: An extractor agent with no access to the taxonomy extracts the atomic technical topics ($u_1, \dots, u_m$) taught or specified in each held-out unit, with verbatim quotes.
- **Pass 2 (Double-Judged Expressibility Matching)**: Two independent read-only judges map each extracted topic $u_j$ against the candidate taxonomy into three grades:
  - `Exact` ($1.0$): Directly represented by a single concept or alias.
  - `Post-coord` ($0.5$): Expressible as a conjunction of $\le 2$ existing concepts (`Method-B` §5.3) without loss of technical specificity.
  - `Missing` ($0.0$): Requires either a broad ancestor catch-all or $>2$ concepts, or is absent.
- **Seeded-Gap Sensitivity Audit (`Method-A` §5.6)**: Before trusting coverage scores, create a shadow copy of the vocabulary with $k$ randomly removed concepts (`seed` recorded) and run Pass 2 on units known to require those concepts. The evaluation protocol is valid only if it flags $\ge 85\%$ of the seeded removals as `Missing` or degraded (`[P]`).

### 5.2 Multi-Rater Indexing Agreement + Specificity Counter-Metrics
On a stratified sample of held-out artifacts indexed independently by two blind indexers:
1. **Subject-level agreement**: Cohen's $\kappa$ on primary subject assignment (`target >= 0.70`).
2. **Concept-set overlap**: Hooper's (1965) indexing consistency $H(A,B) = \frac{|A \cap B|}{|A \cup B|}$ and Rolling's (1981) consistency $R(A,B) = \frac{2|A \cap B|}{|A| + |B|}$ (`target H >= 0.40`, `R >= 0.57`).
3. **Chance-corrected multi-label set agreement**: Krippendorff's $\alpha_{\text{MASI}}$ using Passonneau's (2006) *Measuring Agreement on Set-valued Items* (MASI) distance $d_{\text{MASI}}(A,B) = 1 - J(A,B) \cdot M(A,B)$ without double-squaring (`Method-A` §5.4).
4. **Anti-Gaming Specificity Coupling (`Method-B` §6.7, `Method-C` §6.4)**: Because $\kappa$, Hooper, and Rolling can be artificially inflated by collapsing specific concepts into coarse parents or over-applying root concepts, agreement is valid only when accompanied by:
   - **Leaf/Specific-Concept Ratio**: Fraction of assigned tags that are leaf concepts or depth $\ge 2$ (`must not drop`).
   - **Max Single-Concept Concentration ($p_{\max}$)**: No single concept may account for $>15\%$ of assignments in an area (`Method-B` §6.1).
   - **Effective Vocabulary Utilization**: Fraction of area concepts used at least once on the evaluation corpus.

---

## 6. Traceability Matrix: Synthesized Method vs. Current `taxonomy/METHOD.md`

| Mechanism | Current `taxonomy/METHOD.md` (v0.4-draft) | Synthesized Method | Source Proposal & Rubric Basis |
| :--- | :--- | :--- | :--- |
| **1. Sampling frame & split hygiene** | Ad-hoc syllabus/benchmark checklist; no formal `dev`/`test` split or cluster deduplication | 5-stratum enumerated frame (`S1`–`S5`), `provenance_family_id` cluster split (`dev`/`test`/`reserve`), burned-split rule | `Method-A` §3 (`D3`=5.0), `Method-B` §3 (`D3`=4.25) |
| **2. Two-source warrant independence** | Counts two sources without authorship/derivation deduplication; treats StaffML on par with external warrant | Requires 2 distinct `provenance_family_id` clusters ($\le 50\%$ author overlap, non-derivative); internal questions are leads requiring external corroboration | `Method-A` §2.1 (`D2`=4.75), `Method-B` §2.1–2.2 (`D2`=4.75) |
| **3. Concept-vs-alias & compound splitting** | Editorial judgment (`README.md` notes it is not an operational rule) | Operational 4-part Concept-vs-Alias test + 3-criterion Compound Term Splitting gate + metric$\times$workload factoring | `Method-B` §1 Step 10 & §2.5 (`D2`=4.75), `Method-A` §2.3 (`D2`=4.75) |
| **4. Associative relation discipline** | `related` / `contrasts_with` / `prerequisite` checked for symmetry/acyclicity only | Requires disjointness from `specializes` closure (`S27`) + 1 of 5 typed justification templates | `Method-A` §4.3 (`D4`=4.75), `Method-B` §4.3 (`D4`=4.5) |
| **5. Coverage evaluation** | Single judge; no vocabulary-blind extraction pass; no sensitivity check | Two-pass protocol (Pass 1 blind topic extraction, Pass 2 double-judged matching) + seeded-gap sensitivity audit | `Method-A` §5.6 (`D5`=5.0, `D6`=4.75), `Method-B` §5.3 (`D5`=4.5) |
| **6. Agreement & anti-gaming** | Subject $\kappa$, Hooper, Rolling on 28 questions | Adds $\alpha_{\text{MASI}}$ (Passonneau 2006) + couples agreement with leaf-specificity ratio and max concentration ($p_{\max}$) | `Method-A` §5.4 (`D5`=5.0), `Method-B` §5.2 & §6.7 (`D6`=4.75) |
| **7. Retirement resolution** | Untyped `replaced_by` list | Typed `resolution_class` (`auto`, `broaden`, `review`) + $>20\%$ extension-shift rule | `Method-A` §4.6 (`D4`=4.75), `Method-B` §4.7–4.8 (`D4`=4.5) |
