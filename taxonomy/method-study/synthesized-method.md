# Synthesized Candidate Method for the ML Systems Taxonomy

> **Authoritative enactment status (`2026-11-02`):** `taxonomy/METHOD.md` and the refresh workflow (`.claude/workflows/taxonomy-refresh.md`, enforced mechanically by `reconcile.py`, `apply.py`, `stage_reviewer_workspace.py`, and `./binder/binder concepts check`) constitute the **authoritative enacted specification** for `0.4-draft` and periodic refreshes. This document records how the top-ranked mechanisms from the blind three-family method competition (`Method-A` by `gpt-6-astra`, `Method-B` by `claude-opus-5-5`, and audited scale-stratification elements of `Method-C` by `gemini-4-argon-high`) were synthesized into that enacted protocol, and flags prospective `1.0` schema/governance items where noted.

Every mechanism below cites the blind proposal from which it was drawn, its exact computed cross-family mean score (`dim_means_cross`, with all-judge mean `dim_means_all` in parentheses from `study-receipts-summary.json`) on rubric dimensions `D1`–`D8` (`taxonomy/method-study/rubric.yaml`), and whether each quantitative rule comes from a published standard or a project-calibrated parameter (`[P]`). *Citation provenance note:* ANSI/NISO Z39.19-2005 (R2010) and W3C SKOS (2009) are publicly verifiable; ISO 25964-1:2011 / 25964-2:2013 clause references were screened from model reference knowledge without opening the paywalled ISO text (`METHOD.md`).

---

## 1. End-to-End Construction and Refresh Pipeline (`Stage 1`–`Stage 10`)

**Source provenance**: Synthesized from `Method-B` §1 (14-step DAG with explicit input/output artifacts, stopping rules, and failure recovery; `D1` cross-family mean `4.75/5.0`, all-judge `4.667/5.0`) and `Method-A` §1 (8-stage pipeline with split discipline and pre-registration gates; `D1` cross-family mean `4.00/5.0`, all-judge `4.333/5.0`). Stage identifiers use `Stage 1`–`Stage 10` to avoid colliding with Stack area codes (`S1`–`S4`) or sampling-frame strata (`S1`–`S5`).

| Stage | Name | Inputs | Enacted Outputs / Artifacts | Stopping Condition & Recovery |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | Charter, Scope & Stratified Split Allocation | Domain boundary, enumerated 5-stratum source frame (`S1`–`S5`) | `taxonomy/METHOD.md`, `<campaign>/sampling-manifest.json` (`dev`, `held_out_test`, `reserve` by `provenance_family_id`) | Scope exclusions fixed; `reconcile.py` fails closed if any `provenance_family_id` crosses splits (`Method-A` §3.3). |
| **Stage 2** | Cross-Area Lookup & Candidate Discovery | Staged read-only `dev` workspace (`stage_reviewer_workspace.py`) | `codex-r1.yaml`, `gemini-r1.yaml`, `claude-r1.yaml` with `dev` locators (`workspace-isolation-receipt.json`) | **Incremental refresh (`METHOD.md` Gate 1, `Z39.19 §11.1.1`)**: search all `20` areas in `mlsys-taxonomy-concepts.csv` before proposing new concepts; zero `held_out: true` or `reserve` files staged. *(Blank-slate rebuild mode, tested on Phase 4 `Vocab-X`, masks existing concepts during extraction.)* |
| **Stage 3** | Warrant & Provenance-Lineage Gate | Candidate clusters, `sampling-manifest.json` | Structured `warrant_sources` on warranted items; single-lineage items routed to `gaps:` | Passes enacted Gate 2 (§2.1: $\ge 2$ distinct `dev` `provenance_family_id` lineages across `S1`–`S5`, $\ge 1$ `origin: external`, $\ge 1$ `source_type: prose`; `0` single-source exceptions). |
| **Stage 4** | Product, Benchmark & SKU Exclusion Gate | Warranted candidates & proposed aliases | Filtered candidate set | Passes enacted Gate 3 (§2.2: excludes products, multi-mechanism serving engines, benchmark suites, and SKUs from concepts and multi-mechanism suites from aliases). |
| **Stage 5** | Compound Factoring & `metric × workload` Gate | Multi-word candidates | Pre-coordinated mechanisms vs. post-coordination / rejected `metric × workload` splits | Passes enacted Gate 4 (§2.4: institutionalization across $\ge 2$ `dev` lineages; stage-metric allowlist enforced). |
| **Stage 6** | 4-Part Differential Indexing Witness & Equivalence Gate | Surviving candidates & entry-term proposals | `concepts_add` (`witness_justification`), `add_aliases` (`alias_metadata`), or `gaps:` | Passes enacted Gate 5 (§2.3: 6-key `witness_justification` for concepts; manifest-verified `checked_dev_units` + `scope_note` for `upward-posting` and `quasi-synonym` aliases; unwarranted distinct mechanisms route to `gaps:`, never `aliases`). |
| **Stage 7** | Single Primary Home, Definitions & Typed Relations | Admitted concepts & updates | `patch.yaml` (`home`, non-circular genus-differentia `definition`, `edge_impact_note`, 17-verb `relationships`) | One `home` subject (`home_rule`), `specializes` passes 3-step `all-and-some` (`BTG`) test, `part-of` under `skos:related` (`Z39.19 §8.3.3.2`), $\ge 2$ true relationships per concept. |
| **Stage 8** | 3-Family Cross-Check, Reconciliation & Provenance Sidecar | `patch.yaml`, three `*-r2.yaml` reviews, optional `overrides.yaml` | `final.yaml`, `reconcile-receipt.json` (`schema_version: 3`), `ml-systems-taxonomy.yml`, `taxonomy/term-provenance.yml` | `reconcile.py` enforces $\ge 2$-family reject veto, per-field fix satisfaction, and non-overridable `check_red_flags()`; `apply.py` re-verifies in-process and binds `taxonomy_sha256` in `term-provenance.yml`; 3-family publishability audit (`briefs/audit.md`) passes with `0` blockers. |
| **Stage 9** | Sealed Blind Coverage, Seeded-Gap & Indexing Evaluation | Frozen compiled release, unexposed external `held_out_test` clusters | Coverage & agreement reports (`taxonomy-coverage.json`, `taxonomy-agreement.json`, study receipts) | Two-judge blind coverage + seeded-gap sensitivity audit (§5.1) and dual-tagger agreement ($\kappa$, Hooper, Rolling, $\alpha_{\text{MASI}}$) coupled with leaf-specificity and $p_{\max}$ counter-metrics (§5.2); inspected test clusters are burned globally to `dev`. |
| **Stage 10** | Compiler Verification, Changelog & Release | Compiled artifacts, `mlsysbook-alignment.yml`, `staffml` consumer JSON | Downloadable CSV/JSON/JSON Schema/GraphML/SKOS `.ttl`, `CHANGELOG.md` | `./binder/binder concepts build && check && build --check`, `coverage --check`, `agreement --check`, `pytest`, and `vault check --strict` all exit `0`. |

---

## 2. Concept Admission, Warrant, and Boundary Rules

**Source provenance**: Synthesized from `Method-B` §2 (`D2` cross-family mean `4.50/5.0`, all-judge `4.333/5.0`) and `Method-A` §2 (`D2` cross-family mean `4.50/5.0`, all-judge `4.667/5.0`).

### 2.1 Multi-Lineage Warrant and Heading-Only Ban (Enacted Gate 2)
Following ANSI/NISO Z39.19-2005 §5.3.5 (and conceptually aligned with ISO 25964-1 concept warrant):
1. **Two-Lineage `dev` Warrant Rule (`0` single-source exceptions)**: Every new concept in `concepts_add` requires structured `warrant_sources` from **at least two distinct `provenance_family_id` clusters** in the `dev` split across `S1`–`S5`, including **at least one `origin: external` lineage** (non-`MLSysBook`, non-`Harvard CS249r`, non-`StaffML`) and **at least one `source_type: prose` unit** containing explanatory text defining the mechanism (`reconcile.py`). While `Method-B` §2.1 explored a single-source curriculum/standard exception, the enacted protocol (`METHOD.md` Gate 2) allows **zero single-source concept admissions**: single-lineage mechanisms are recorded in `gaps:` until a second independent lineage appears.
2. **Authorship & Derivation Clustering**: Two sources merge into a single `provenance_family_id` cluster (counting as one lineage, whereas distinct lineages have `< 50%` overlap) when:
   - Their first/senior author sets overlap by $\ge 50\%$ (`|A_i \cap A_j| / \min(|A_i|, |A_j|) \ge 0.50`, transitive closure; `Method-A` §2.1, `[P]`),
   - One work is a companion repository, documentation site, lecture adaptation, or tutorial authored by the creators of the other, or
   - Both belong to project-internal corpora (`pf_mlsysbook_*`, `pf_harvard_cs249r_*`, `pf_staffml_*` — all tagged `origin: internal`, plus single-lab Editor-primary/senior papers in prospective refreshes).
3. **Heading-Only Warrant Ban**: A syllabus or venue-checklist topic title (`source_type: heading`) may serve as the corroborating second lineage when paired with a `source_type: prose` source that defines the mechanism, but bare headings alone (`0` prose sources) cannot mint a concept or justify extrapolated mechanisms.

### 2.2 Named Entity, Product, Benchmark, and SKU Exclusion Gate (Enacted Gate 3)
Commercial products, multi-mechanism serving engines/frameworks, hardware SKUs, model releases, and benchmark suites (`vLLM`, `TensorRT-LLM`, `SGLang`, `DeepSpeed`, `PyTorch`, `H100`, `B200`, `MI300X`, `MLPerf`, `benchmark suite`) are **excluded as concepts** (`Z39.19 §6.3.3` project scope policy; `Method-A` §2.2, `Method-B` §2.4):
- **Mechanism extraction**: Extract the underlying architectural mechanism (`vLLM` $\rightarrow$ `mlsys.paged-kv` and `mlsys.continuous-batching`; `ZeRO` $\rightarrow$ `mlsys.zero-sharding`).
- **Alias restriction**: Only a named algorithm or single-mechanism implementation that is 1-to-1 synonymous with one mechanism (`PagedAttention` on `mlsys.paged-kv`, `ZeRO` and `FSDP` on `mlsys.zero-sharding`, `CUDA Graphs` on `mlsys.graph-capture-replay`) may be added as an `alias`. Multi-mechanism frameworks and benchmark suites (`vLLM`, `TensorRT-LLM`, `SGLang`, `MLPerf`, `benchmark suite`) are excluded from both concepts and aliases (`MULTI_MECHANISM_SUITE_ALIASES` in `reconcile.py`).
- **No `BTI` (`instance-of`) relation**: Individual named standards or open interchange formats enter only as general architectural mechanisms when warranted across $\ge 2$ lineages; `BTI` is excluded from the relation schema.

### 2.3 Operational 4-Part Concept-vs-Alias Witness & `Z39.19 §8.2` Equivalence Gate (Enacted Gate 5)
To decide whether candidate $C$ is a separate `Concept`, a typed `alias` on existing concept $P$, or a `gaps:` entry (`Method-A` §2.3, `Method-B` §1 Step 10 & §2.5, enacted in `METHOD.md` Gate 5):
1. **Criterion (a) — Distinct System Invariant / Envelope**: Does $C$ introduce a distinct system invariant, resource envelope, failure boundary, analytical causal model, or measurement semantics not already fixed by $P$?
2. **Criterion (b) — Independent Decision Consequence / Trade-Off**: Does $C$ carry an independent failure mode, trade-off, or engineering decision consequence not identical to $P$?
3. **Criterion (c) — Taxonomic / Causal Profile**: Is $C$ either a valid `all-and-some` (`specializes`) child of $P$ with its own differentiating properties or an endpoint of a causal/associative edge not inherited from $P$?
4. **Criterion (d) — Structured Differential Indexing Witness & Two-Lineage Warrant**: Can the proposer supply a 6-key `witness_justification` (`compared_concept`, `criterion_a`/`b`/`c`, and distinct `scenario_a` and `scenario_b`) **and** $\ge 2$-lineage `dev` warrant (`Gate 2`)?

**Enacted three-way decision & tiebreak rule**:
- **Separate `Concept`**: Admitted when Criteria (a)–(d) and Gate 2 ($\ge 2$ `dev` lineages) all hold.
- **Unwarranted distinct mechanism $\rightarrow$ `gaps:` (never `aliases`)**: When a candidate denotes a substantively distinct mechanism or methodology under Criteria (a)–(c) (for example, `adaptive stress testing`) **but lacks two-lineage `dev` warrant**, it routes to `gaps:`, **never** to `aliases`. Likewise, constituent execution phases (`prefill phase` vs. `decode phase`), distinct analytical methodologies (`FMEA`, `FTA`, `STPA` under `Hazard analysis`), planning artifacts (`preregistered analysis plan`), and method-produced artifacts (`unsafe control action` under `STPA`) are never aliased onto a parent.
- **Typed `alias` (`synonym`, `quasi-synonym`, `acronym`, or `< 2`-lineage `upward-posting`)**: Permitted only when the surface term is verbatim-attested, qualified against cross-area homographs (`Z39.19 §5.3.1, §6.3.2`, e.g., `dynamic request batching` on `mlsys.batching`), and either denotes the same mechanism (`synonym` / `acronym` / `Z39.19 §8.2.3` `quasi-synonym` such as `selective abstention` on `mlsys.selective-prediction`) or is a strict `all-and-some` narrower variant (`Z39.19 §8.2.4` `upward-posting` such as `grammar-constrained decoding` on `mlsys.constrained-decoding`) with **`dev_lineages_requiring_separation` $\in \{0, 1\}$ (`< 2` `dev` lineages)**. Both `upward-posting` and `quasi-synonym` require manifest-verified `checked_dev_units` and `scope_note` in `reconcile.py`.

### 2.4 Compound Term Factoring vs. Pre-Coordination (Enacted Gate 4, `Z39.19 §7`)
A multi-word compound term $C$ is admitted as a pre-coordinated concept only when (`Method-A` §2.3, `Method-B` §2.5):
1. It names a lexicalized algorithm, protocol, or system mechanism with non-compositional behavior or its own failure mode/trade-off (`speculative decoding`, `continuous batching`, `activation checkpointing`), **and**
2. It is institutionalized as a unit across **$\ge 2$ distinct `provenance_family_id` lineages** in `dev` (`[P]`, aligned with Gate 2 in `METHOD.md`; `Method-A` originally proposed $\ge 3$ before calibration to $\ge 2$).

Generic **`metric × workload`** combinations (`inference throughput`, `inference energy cost`, `training power`) are factored into post-coordination rather than admitted, except for the explicit stage-specific structural metric allowlist (`TTFT`, `TPOT`, `inter-token latency`, `goodput`, `MFU`, `model bandwidth utilization`, `request throughput`, `latency percentiles`).

---

## 3. Enumerated Sampling Frame & Split Hygiene

**Source provenance**: Synthesized from `Method-B` §3 (`D3` cross-family mean `4.75/5.0`, all-judge `4.667/5.0`), `Method-A` §3 (`D3` cross-family mean `4.50/5.0`, all-judge `4.667/5.0`), and `Method-C` §3.2 (`D3` cross-family mean `3.75/5.0`, all-judge `4.00/5.0` — scale representation across cloud, enterprise/workstation, and edge/TinyML). Full details live in `taxonomy/method-study/sampling-frame.md`.

1. **Five Stratified Source Populations (`S1`–`S5`)**: `S1` Textbooks & Reference Monographs, `S2` University Courses, `S3` Peer-Reviewed Systems & ML Venues, `S4` Practitioner Artifacts & Diagnostic Scenarios, and `S5` Standards, Assurance Guidance & Benchmark Specifications.
2. **Whole-Cluster Split Partition (`<campaign>/sampling-manifest.json`)**:
   - Every unit declares `unit_id`, `provenance_family_id`, `stratum` (`S1`–`S5`), `split` (`dev`, `held_out_test`, `reserve`), `origin` (`external` | `internal`), and `source_type` (`prose` | `heading`).
   - Prospective refreshes target `60%` `dev`, `25%` `held_out_test`, and `15%` `reserve` at the whole-cluster level (`0` cross-split clusters; diagnostic pre-checks use separate one-off slices burned to `dev` upon inspection per `taxonomy-refresh.md` Step 1b).
3. **Global Cluster-Level Burn Rule**:
   - Once any unit in a `held_out_test` slice is inspected to diagnose gaps or compare vocabularies, **every unit sharing its `provenance_family_id` cluster across all areas** is promoted to `dev` (`held_out: false` on venue checklists) and may never again be cited as an unexposed held-out test.

---

## 4. Structural Invariants, Relationships, and Lifecycle Governance

**Source provenance**: Synthesized from `Method-B` §4 (`D4` cross-family mean `4.50/5.0`, all-judge `4.667/5.0`) and `Method-A` §4 (`D4` cross-family mean `4.25/5.0`, all-judge `4.50/5.0`).

### 4.1 Polyhierarchy with Single Primary Home (`home`)
- Every concept declares exactly one primary subject row (`home`), forming a clean partition across `101` subjects in `20` areas (`Method-A` §4.1, `Method-B` §4.2). Cross-area relevance uses `secondary_concepts` with an explicit `rationale` (`reconcile.py`).
- Concept-to-concept `specializes` (`rdfs:subPropertyOf skos:broader`) forms a cycle-free DAG verified by the 3-step **generic (`BTG`) `all-and-some` test** (`Z39.19 §8.3.1`). Because ML system components participate in multiple assemblies, `part-of` is modeled as a cycle-checked associative relation exported under `skos:related` (`Z39.19 §8.3.3.2`), and `BTI` is not used.

### 4.2 Controlled 17-Verb Relationships & Incident-Edge Preservation
- Relationships use the closed 17-verb LinkML vocabulary (`specializes` + 16 associative/causal/evaluation/composition verbs under `skos:related`), disjoint from `specializes` transitive closure (`SKOS S27` / `Z39.19 §8.4`), with mandatory `predicate` and `conditions` strings verified in both forward and inverse literal readings.
- Redefining an existing concept requires an explicit `edge_impact_note` (`reconcile.py` & `term-provenance.yml`) verifying that every incident relationship remains literally true without narrowing or shifting extension.

### 4.3 Non-Circular Genus–Differentia Definitions & Compiler Gates
- Every concept `definition` is a single non-circular genus-differentia sentence stating what kind of mechanism, structure, failure mode, or metric it is and what it changes in the system (`Z39.19 §6.2.2`; post-patch `(effective_label, effective_defn)` circularity enforced by `reconcile.py`).
- `./binder/binder concepts check` enforces LinkML schema conformance, `SKOS S14` (`prefLabel` cardinality) plus global case-folded preferred/entry-term uniqueness across active and retired concepts (`Z39.19 §5.3.1`), `SKOS S13` (`prefLabel`/`altLabel` disjointness), hierarchical DAG acyclicity, polarity consistency, and `taxonomy/term-provenance.yml` SHA-256 binding (`pytest`).

### 4.4 Permanent Identities, Split Consistency, and Retirement Schema
Concept IDs (`mlsys.*`) are permanent and never reused (`Method-A` §4.6, `Method-B` §4.7–4.8, `METHOD.md`):
- **Label refinement (unchanged extension)**: Update `label` and retain the former label in `aliases`.
- **1-to-N concept split rule (unified with `METHOD.md`)**: When a coarse concept is split into narrower mechanisms and one mechanism clearly retains the dominant share of existing book/question references without contradicting historical tags, retain the existing `mlsys.*` ID on that dominant half and mint new `mlsys.*` ID(s) for the split-off mechanism(s), re-auditing downstream tags. When a split alters extension symmetrically ($>20\%$ bilateral shift `[P]`) such that neither child preserves existing tags cleanly, retire the old ID (`concepts_retire`) pointing `replaced_by` to the primary replacement and re-tag downstream consumers explicitly.
- **Enacted `0.4-draft` `RetiredConcept` vs. prospective `1.0` fields**: In `0.4-draft` (`mlsys-taxonomy.linkml.yml` and `apply.py`), `RetiredConcept` persists `id`, former `label`, former `aliases`, transitive-resolved active `replaced_by`, `version`, and `reason`. Adding machine-typed `resolution_class` (`auto`, `broaden`, `review`) and multivalued `split_into` slots is recorded as a **prospective `1.0` LinkML schema extension** for external board review (`candidate-experts.md`).

---

## 5. Evaluation Protocol, Empirical Calibration Findings, and Anti-Gaming Controls

**Source provenance**: Synthesized from `Method-B` §5–6 (`D5` cross-family mean `4.75/5.0` [all-judge `4.667`], `D6` `4.75/5.0` [`4.667`], `D7` `4.75/5.0` [`4.667`]) and `Method-A` §5–6 (`D5` `4.00/5.0` [`4.167`], `D6` `4.25/5.0` [`4.333`], `D7` `4.00/5.0` [`4.167`]).

### 5.1 Two-Pass Blind Coverage & Phase 6 Seeded-Gap Sensitivity Finding
To test vocabulary completeness on unexposed `origin: external` `held_out_test` clusters (`Method-A` §5.6, `Method-B` §5.3):
- **Pass 1 (Vocabulary-Blind Topic Extraction)**: Extract central mechanisms from external held-out units in an isolated directory with no access to the taxonomy.
- **Pass 2 (Two-Family Expressibility Scoring)**: Two distinct model families score each unit (`Exact = 1.0` single concept/alias, `Post-coord = 0.5` conjunction of `2`–`3` concepts without specificity loss, `Missing = 0.0`), reporting both `any-judge` and conservative `consensus` (`both == 1.0`) binary coverage, binary coverage $\kappa$, and `10,000`-resample cluster-bootstrap 95% CIs by `provenance_family_id`.
- **Seeded-Gap Sensitivity Audit & Empirical Phase 6 Limitation (`Method-A` §5.6)**: `Method-A` proposed removing $k$ concepts (`k = 10` in Phase 6) and requiring Pass 2 to degrade $\ge 85\%$ (`[P]`) of baseline-covered judgments citing those concepts. **Empirical Phase 6 result (`study-receipts-summary.json`):** unrestricted LLM judges degraded only **`24 / 40 = 60.0%`** of affected judgments on `Vocab-Y` and **`16 / 29 = 55.2%`** on `Vocab-X` — **failing the $\ge 85\%$ sensitivity target** because judges compensated for deleted leaf concepts by substituting broader parent/neighbor concepts or post-coordination (`0.5`). Consequently, `METHOD.md` and `taxonomy-refresh.md` Step 5b treat unpinned LLM binary coverage as over-inclusive on fine-grained deletions, rely on conservative two-judge `Exact` (`1.0`) consensus plus pinned concept-ID verification, and mandate reporting seeded-gap sensitivity alongside every held-out coverage run.

### 5.2 Multi-Rater Indexing Agreement + Specificity Counter-Metrics
On held-out scenarios indexed independently by two blind taggers (`1` primary subject, `0`–`3` concepts):
1. **Primary subject agreement**: Cohen's $\kappa$ (`release floor >= 0.70` LLM, `>= 0.65` 12-human pre-`1.0` study).
2. **Concept-set consistency**: Both conditional (`both non-empty`) and all-item ($J(\emptyset,\emptyset)=1$, $J(A,\emptyset)=0$) **Hooper (1965)** and **Rolling (1981)** overlap (`LLM floors >= 40% / >= 57%` conditional, `>= 35% / >= 50%` all-item), paired with empty-assignment rate ($\le 20\%$).
3. **Chance-corrected set agreement**: **Krippendorff's $\alpha_{\text{MASI}}$** using Passonneau's (2006) MASI distance $d_{\text{MASI}}(A,B) = 1 - J(A,B)\cdot M(A,B)$ (`LLM floor >= 0.40`).
4. **Anti-Gaming Specificity Counter-Metrics (`Method-B` §6.7, `Method-C` §6.4)**:
   - **Leaf-concept specificity ratio**: Share of assigned concept tags whose concept has `0` incoming `specializes` children (in Phase 6 `IDX`, `Vocab-Y` = `148 / 183 = 80.9%`, `Vocab-X` = `186 / 216 = 86.1%`).
   - **Max single-concept share ($p_{\max}$)**: $\le 15.0\%$ (`Vocab-Y` = `9.84%`, `Vocab-X` = `6.94%`).
   - **Effective vocabulary utilization**: Distinct concepts assigned divided by active area concepts (`Vocab-Y` = `44 / 125 = 35.2%`, `Vocab-X` = `55 / 111 = 49.5%`).

---

## 6. Traceability Matrix: Pre-Study Baseline vs. Synthesized Design vs. Enacted `METHOD.md`

*Rubric scores below are exact cross-family means (`dim_means_cross`, excluding self-ratings; all-judge means in `study-receipts-summary.json`: `Method-B` `37.25/40` cross / `37.00/40` all, `Method-A` `34.00/40` cross / `35.33/40` all, `Method-C` `25.75/40` cross / `29.83/40` all).*

| Mechanism | Pre-study `0.4-draft` `METHOD.md` (before Phase 7) | Enacted Post-Study `METHOD.md` & `reconcile.py` | Blind Proposal & Exact Cross-Family Score |
| :--- | :--- | :--- | :--- |
| **1. Sampling frame & split hygiene** | Checklist-level `held_out` flag only; no `provenance_family_id` clustering or `reserve` pool | 5-stratum frame (`S1`–`S5`), whole-cluster `dev`/`held_out_test`/`reserve` in `sampling-manifest.json`, `stage_reviewer_workspace.py` `dev` isolation, global cluster burn | `Method-B` §3 (`D3 = 4.75`), `Method-A` §3 (`D3 = 4.50`), `Method-C` §3.2 (`D3 = 3.75`) |
| **2. Two-lineage warrant & prose check** | Two sources without authorship clustering or internal/external separation | $\ge 2$ distinct `dev` `provenance_family_id` clusters (`< 50%` author overlap), $\ge 1$ `origin: external`, $\ge 1$ `source_type: prose`, `0` single-source exceptions | `Method-A` §2.1 (`D2 = 4.50`), `Method-B` §2.1–2.2 (`D2 = 4.50`) |
| **3. Concept-vs-alias & compound gates** | Unstructured editorial judgment; `MLPerf` / `benchmark suite` / `preregistered analysis plan` leaked into aliases | 6-key `witness_justification`, manifest-checked `alias_metadata` (`upward-posting` `< 2` `dev` lineages & `quasi-synonym`), $\ge 2$-lineage compound gate, unwarranted concepts $\rightarrow$ `gaps:` | `Method-B` §1 & §2.5 (`D2 = 4.50`), `Method-A` §2.3 (`D2 = 4.50`) |
| **4. Structural & provenance gates** | 17 LinkML verbs + `SKOS S27`; no committed provenance sidecar binding `ml-systems-taxonomy.yml` | Adds `edge_impact_note`, `remove_aliases_rationale`, former `label`/`aliases` on `RetiredConcept`, in-process `verify_campaign` in `apply.py`, and SHA-256-bound `taxonomy/term-provenance.yml` (`resolution_class`/`split_into` queued for `1.0`) | `Method-B` §4 (`D4 = 4.50`), `Method-A` §4.3 & §4.6 (`D4 = 4.25`) |
| **5. Coverage & seeded-gap audit** | Single-judge checklist status without seeded-gap sensitivity calibration | Two-family blind scoring + cluster-bootstrap CIs + mandatory seeded-gap sensitivity audit (disclosing Phase 6 `60.0%` / `55.2%` shortfall vs. `>= 85%` target) | `Method-B` §5.3 (`D5 = 4.75`, `D6 = 4.75`), `Method-A` §5.6 (`D5 = 4.00`, `D6 = 4.25`) |
| **6. Agreement & specificity coupling** | Subject $\kappa$ and conditional Hooper/Rolling on `0.2-draft` StaffML samples | Adds all-item Hooper/Rolling, $\alpha_{\text{MASI}}$, empty-set ceiling ($\le 20\%$), leaf-concept ratio (`0` `specializes` children), $p_{\max} \le 15\%$, and `240`-item / `40`-anchor 12-human `1.0` protocol | `Method-B` §5.2 & §6.7 (`D5 = 4.75`, `D6 = 4.75`), `Method-A` §5.4 (`D5 = 4.00`), `Method-C` §6.4 (`D6 = 3.00`) |
