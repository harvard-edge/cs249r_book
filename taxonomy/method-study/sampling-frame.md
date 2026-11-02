# Enumerated Sampling Frame and Split Hygiene for the ML Systems Taxonomy

This document specifies the five-stratum sampling frame, provenance-family clustering rules, and split allocation protocol (`dev`, `held_out_test`, and `reserve`) synthesized from `Method-A` (§3), `Method-B` (§3), and audited `Method-C` (§3.2), together with the executed Phase 3 allocation (`taxonomy/method-study/preregistration.yaml`) and post-Phase-6 global cluster-burn state for `P4` and `Q4`.

*Note: Sampling-frame strata `S1`–`S5` denote source populations and are distinct from Stack area codes `S1`–`S4`.*

---

## 1. Five Stratified Source Populations (`S1`–`S5`)

Every candidate concept and every coverage or indexing evaluation unit is drawn from an enumerated frame partitioned into five source populations (`Method-A` §3.1, `Method-B` §3.1). Source stratum records **where an excerpt comes from**; warrant type under ANSI/NISO Z39.19 §5.3.5.1–§5.3.5.3 is recorded separately so operational proxies are not conflated with direct user search logs or organizational term-preference mandates:

| Stratum ID | Stratum Name | Unit of Sampling | Inclusion Criteria & Warrant Role |
| :--- | :--- | :--- | :--- |
| **`S1`** | **Textbooks & Reference Monographs** | Chapter / major section | Published or open-access textbooks covering ML systems architecture, distributed training/serving, hardware acceleration, or safety/reliability engineering (both external monographs and project-internal `MLSysBook` chapters). Provides **literary & curriculum warrant** (`Z39.19 §5.3.5.1`; internal `MLSysBook` chapters carry `origin: internal`). |
| **`S2`** | **University Courses** | Course syllabus (lecture module + assignment/lab scope) | Graduate or upper-division undergraduate ML systems courses at accredited universities with public lecture schedules and reading lists, stratified across hardware/compilers, systems/networking, and safe/reliable deployment (`Method-A` §3.2). Provides **external literary & pedagogical warrant** (except Harvard `CS249r`, which shares `MLSysBook` authorship and carries `origin: internal`). |
| **`S3`** | **Peer-Reviewed Systems & ML Venues** | Paper (title, abstract, architecture/evaluation passage, or blind venue topic entry) | Peer-reviewed papers from systems conferences (`MLSys`, `OSDI`, `SOSP`, `NSDI`, `ASPLOS`, `ISCA`, `EuroSys`, `CAV`, `AISec`) and systems/safety tracks of ML/NLP venues (`NeurIPS`, `ICML`, `ICLR`, `EMNLP`, `FAccT`, `SATML`) within the six-calendar-year study window (`2021`–`2026` inclusive at study freeze) plus pre-registered foundational papers (`>500` citations in Google Scholar / OpenAlex or standard `S2` syllabus adoption). Provides **external literary warrant**. |
| **`S4`** | **Practitioner Artifacts, Postmortems & Diagnostic Scenarios** | Engineering design doc, incident postmortem, runtime spec, or diagnostic scenario | Public production engineering design guides, open-source serving/runtime documentation (`Triton`, `Ray Serve`, `ONNX Runtime` — used solely to extract underlying mechanisms, never product concepts), frontier deployment governance policies, and `StaffML` diagnostic scenarios (`origin: internal`). Provides **operational literary warrant** and (for `StaffML` tag counts across the `10,716`-question bank) an **assignment-frequency proxy**, not direct user search/request warrant (`Z39.19 §5.3.5.3`). |
| **`S5`** | **Standards, Assurance Guidance & Benchmark Specifications** | Standard clause / assurance guidance module / benchmark scenario specification | Published technical standards, public assurance guidance reports, and benchmark specifications (`NIST AI 100-1`, `NIST AI 600-1`, `ISO 26262`, `ISO 21448`, `ISO/IEC 42001`, `ISO/PAS 8800`, `UL 4600`, `OWASP Top 10 for LLM`, `MITRE ATLAS`, `MLPerf Inference/Training/Tiny`). Provides **normative literary warrant** (`Z39.19 §5.3.5.1`), not internal organizational preferred-term mandates (`Z39.19 §5.3.5.2`) unless term-form adoption is separately logged. |

### Cross-Stratum Scale Representation (`Method-C` §3.2, `Method-B` §3.2)
To prevent datacenter LLM serving from crowding out resource-constrained or classical ML systems, the frame requires representation across execution scales where applicable to an area:
- **Datacenter / Cloud Scale** (multi-node accelerator cluster serving, training, and fleet scheduling)
- **Enterprise / Single-Node / Workstation** (gateway/microservice orchestration, CPU/single-accelerator inference, retrieval and compound pipelines)
- **Edge / On-Device / Embedded (`TinyML`)** (MCU/NPU/mobile quantization, static arena memory planning, battery/thermal envelopes)

---

## 2. Global Provenance-Family Clustering (`provenance_family_id`)

Before stratification, split assignment, or warrant counting, every unit in `<campaign>/sampling-manifest.json` receives an explicit `unit_id`, `provenance_family_id`, `stratum` (`S1`–`S5`), `split` (`dev`, `held_out_test`, `reserve`), `origin` (`external` | `internal`), and `source_type` (`prose` | `heading`) (`Method-A` §2.1 & §3.3, `Method-B` §2.2 & §3.4):

1. **Shared Authorship / Lab Lineage (Deterministic Overlap & Transitive Closure)**: Let $A_i$ denote the set of first and last (primary and senior) authors of external source $i$. Two sources $i, j$ are linked when $|A_i \cap A_j| / \min(|A_i|, |A_j|) \ge 0.50$ (so distinct lineages have `< 50%` overlap), and connected components (transitive closure) share a single global `provenance_family_id` across all strata and areas.
2. **Paper–System–Benchmark–Checklist Lineage**: A research paper, its companion open-source documentation, tutorials derived from it, and any checklist row summarizing that same work share one `provenance_family_id` (for example, Kwon et al. SOSP 2023 PagedAttention + vLLM architecture docs = one cluster; Huyen 2022 prose excerpts + `courses-and-standards-2026` `DMLS` topic rows = `pf_huyen_dmls`).
3. **Project-Internal Corpora & Editor-Authorship Policy (`origin: internal`)**:
   - **Gate 2 external-warrant rule**: All units drawn from `MLSysBook` chapters (`pf_mlsysbook_*`), `Harvard CS249r` course materials (`pf_harvard_cs249r_*`), and `StaffML` diagnostic banks (`pf_staffml_*`) carry `origin: internal` (`INTERNAL_LINEAGE_PREFIXES` in `reconcile.py`). In prospective refreshes across new areas, single-lab papers where the taxonomy Editor is a primary or senior author likewise carry `origin: internal` and are assigned to `dev` (multi-institutional consortium specifications such as `MLPerf Inference` remain `S5` normative standards, while `MLPerf` itself remains excluded from concepts and aliases under Gate 3). Any number of `origin: internal` units supplies `0` external lineages under Gate 2.
   - **Historical Phase 3 internal sub-unit exception & StaffML holdout trace**: In the executed Phase 3 `P4`+`Q4` study (`preregistration.yaml`), internal units were sub-partitioned by chapter (`pf_mlsysbook_<vol>/<chapter>`) and by `StaffML` sample pool (`pf_staffml_dev_*`, `pf_staffml_heldout_*`) with `0` `provenance_family_id` clusters crossing splits, and then **excluded from external generalization claims**:
     1. All `7` held-out `MLSysBook` chapter units were excluded from Phase 6 coverage (`COV` evaluated only `79` `origin: external` `S2`/`S3`/`S5` units).
     2. Direct audit of the `40` `S4` `StaffML` `held_out_test` indexing units (`38` in `taxonomy/agreement/staffml-300-heldout-2026-10.yml` + `2` in `staffml-300-2026-10.yml`, originally tagged at `0.2-draft` [`714`/`727` concepts]) against `staffml/vault/data/taxonomy-tagging-2026-10.yaml` confirms that **`0 / 40` (`0/38` + `0/2`) appeared in `proposed_concepts`, `subject_only_questions`, or `adjudication`**, while **`13 / 40` (`12/38` + `1/2`) appeared in `review_queue`** of the Oct 2026 whole-bank (`10,716`-question) tagging pass. Accordingly, those `38` items are reported solely as a **tagging-rule calibration holdout** (`tagging_rule_holdout_38_staffml_2026_10` in `study-receipts-summary.json`), whereas **only external 2026 `systems-venues-2026` units were unexposed to `0.4-draft` construction**. In post-Phase-6 campaigns (`bucket-a-campaign/sampling-manifest.json`) and prospective refreshes, all `MLSysBook`, `CS249r`, and `StaffML` clusters reside in `dev`.

**Mechanical enforcement (`reconcile.py` & `stage_reviewer_workspace.py`)**:
- `reconcile.py` requires `<campaign>/sampling-manifest.json` whenever `concepts_add` or `upward-posting` / `quasi-synonym` aliases appear, rejects duplicate `unit_id`s or missing `split`/`origin`/`source_type` fields, and fails closed if any `provenance_family_id` crosses splits. `stage_reviewer_workspace.py` stages only `held_out: false` checklists and `split: "dev"` manifest units/chapters for Round-1 and Round-2 LLM reviewers.

---

## 3. Split Allocation Protocol and Post-Phase-6 Cluster-Burn State

1. **Prospective Refresh Targets (`60%` / `25%` / `15%`)**:
   - In periodic refreshes (`.claude/workflows/taxonomy-refresh.md`, `taxonomy/METHOD.md`), global `provenance_family_id` clusters are assigned once before construction with target cluster proportions of **`60%` `dev`** (visible to builders), **`25%` `held_out_test`** (sealed external clusters for Step 5b evaluation), and **`15%` `reserve`** (sequestered external clusters for replenishing burned `held_out_test` slices alongside fresh post-cutoff external sources). Multi-area clusters are assigned as whole clusters prior to per-area tallies.
2. **Executed Phase 3 Allocation in the Two-Area Study (`preregistration.yaml`, `seed = 20261103`)**:
   - In the completed Phase 3 `P4`+`Q4` study (`382` units across `97` unique `provenance_family_id` clusters; `0` cross-split clusters), `build_phase3_preregistration.py` combined pre-designated benchmark splits (`fixed_split`) with a deterministic cell-wise draw (`seed = 20261103`) over movable `(area, stratum)` cells of size $n$ (`1` cluster to `held_out_test` when $n \ge 3$, `1` cluster to `reserve` when $n \ge 4$, and the remaining $n - \mathbb{I}(n \ge 3) - \mathbb{I}(n \ge 4)$ clusters to `dev`).
   - Because single-passage curated sources each formed a 1-unit `dev` cluster while `held_out_test` and `reserve` held larger venue/question-bank clusters, the **achieved Phase 3 cluster split was `76 / 97` (`78.4%`) `dev`, `15 / 97` (`15.5%`) `held_out_test`, and `6 / 97` (`6.2%`) `reserve`**, and the **achieved unit split was `200 / 382` (`52.4%`) `dev`, `126 / 382` (`33.0%`) `held_out_test`, and `56 / 382` (`14.7%`) `reserve`**:
     - **Clusters by split (`97` total in Phase 3)**: `76` `dev` (`47` `P4`-only, `23` `Q4`-only, `6` `P4`/`Q4`-shared; `45` `external`, `31` `internal`), `15` `held_out_test` (`8` `P4`-only, `6` `Q4`-only, `1` `P4`/`Q4`-shared; `10` `external`, `5` `internal`), and `6` `reserve` (`3` `P4`-only, `3` `Q4`-only; `4` `external`, `2` `internal`).
     - **Units by split, area, and origin (`382` total in Phase 3)**: `dev` = `200` (`142` `P4` [`95` `external`, `47` `internal`], `58` `Q4` [`14` `external`, `44` `internal`]); `held_out_test` = `126` (`116` `P4` [`75` `external`, `41` `internal`], `10` `Q4` [`4` `external`, `6` `internal`]); `reserve` = `56` (`47` `P4` [`46` `external`, `1` `internal`], `9` `Q4` [`2` `external`, `7` `internal`]).
3. **Post-Phase-6 Global Cluster-Burn State (`bucket-a-campaign/sampling-manifest.json`)**:
   - Inspecting the `15` Phase 6 `held_out_test` clusters (`126` units) during the `2026-11-02` comparison burned them globally, including **all `4` venue clusters (`osdi2026`, `nsdi2026`, `eurosys2026`, `asplos2026`) of `taxonomy/coverage/systems-venues-2026.yml`** (now marked `held_out: false` across the entire repository).
   - Furthermore, while `MLSys 2026` (`pf_cov_mlsys_25_26_mlsys2026`, `45` units from `mlsys-papers-2025-2026.yml`), `pf_cov_courses_2026_USC` (`1` unit from `courses-and-standards-2026.yml`), and `8` `MLSysBook` chapter units were unexposed to the blind `Vocab-X` rebuild during Phase 6, they came from `held_out: false` checklists or internal `MLSysBook` chapters already exposed to `0.4-draft` (`Vocab-Y`).
   - Reconciling work-level duplicate cluster IDs (`pf_cov_courses_2026_DMLS -> pf_huyen_dmls` and `pf_cov_courses_2026_NIST -> pf_nist_ai_rmf`), adding the verbatim prose excerpt `P4-S1-MLSYSBOOK-V1C05` (`books/vol1/05_nn_computation/05_nn_computation.qmd:4552`), and promoting all Phase-6-burned and `0.4-draft`-exposed clusters to `dev` yields **`383` units across `95` deduplicated clusters in `bucket-a-campaign/sampling-manifest.json`: `381` `dev` units (`93` clusters), `0` `held_out_test` units, and `2` unexposed external `reserve` units (`pf_ut_cs378` [`P4-S2-UT378-01`] and `pf_iso42001_pas8800` [`Q4-S5-ISO42001-PAS8800`])**. Replenishing `held_out_test` for the next refresh therefore requires fresh post-cutoff external clusters (such as `MLSys 2027` and `2027` systems venues), never `mlsys-papers-2025-2026.yml` or `systems-venues-2026.yml`.

---

## 4. Executed Two-Area Sampling Frame (`P4` and `Q4` — Exact `382`-Unit Phase 3 Inventory)

In Phase 3 (`2026-11-02`), the `382`-unit manifest (`random_seed: 20261103`) was committed in `taxonomy/method-study/preregistration.yaml` (enumerating `S1`–`S5` source pools for the remaining `18` taxonomy areas remains open pre-`1.0` work). Every item listed below corresponds directly to executed units in `phase4-dev-sources.json` and `phase6-heldout-slice.json`:

### Area 1: `P4` (`serving`) — Deployment & serving (`9` subjects, `95` `0.4-draft` concepts)
Subjects (`9`): `P4.1` (`serving-architecture`) Serving architectures · `P4.2` (`compound-ai`) Compound AI systems & model routing · `P4.3` (`latency-goodput`) Latency, throughput & goodput · `P4.4` (`request-scheduling`) Batching, admission & autoscaling · `P4.5` (`llm-inference`) LLM & generative inference · `P4.6` (`structured-generation`) Decoding, sampling & structured output · `P4.7` (`test-time-compute`) Test-time compute · `P4.8` (`retrieval-rag`) Retrieval & vector search · `P4.9` (`on-device-deployment`) On-device & edge deployment.

- **`S1` (Textbooks & Monographs — `50` `dev`, `5` `held_out_test`, `1` `reserve` in Phase 3)**:
  - `P4-S1-DMLS-01` & `P4-S1-DMLS-02` (`dev`): Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022) — Ch. 7–8 (batch vs. online prediction, dynamic batching, compression/export).
  - `P4-S1-WSC-01` (`dev`): Barroso, Hölzle, & Ranganathan, *The Datacenter as a Computer* (3rd ed., 2018; pre-2021 foundational monograph) — Ch. 2 & 6 (tail latency, hedging, admission control).
  - `47` `dev` (+ `P4-S1-MLSYSBOOK-V1C05` prose excerpt added in Phase 7), `5` `held_out_test` (`vol1/13_model_serving`, excluded from Phase 6 `COV`), and `1` `reserve` (`vol1/06_nn_architectures`) `MLSysBook` Vol. I–IV chapter units (`origin: internal`; all promoted to `dev` post-Phase 6).
- **`S2` (University Courses — `33` `dev`, `2` `held_out_test`, `1` `reserve` in Phase 3)**:
  - Curated syllabi: `P4-S2-CMU11868-01` & `P4-S2-CMU11868-02` (CMU `11-868` *Large Language Model Systems*, Spring 2025, `dev`), `P4-S2-MIT65940-01` (MIT `6.5940` *TinyML and Efficient Deep Learning Computing*, Fall 2024, `dev`), `P4-S2-STANFORD-CS336-01` (Stanford `CS336` / `CS229S`, 2025, Phase 6 `held_out_test`), and `P4-S2-UT378-01` (UT Austin `CS 378`, Fall 2024, unexposed external `reserve`).
  - Checklist topic units from `taxonomy/coverage/courses-and-standards-2026.yml` (`30` `dev`, `1` `held_out_test` [`P4-COV-courses-and-standards-2026-212`]).
- **`S3` (Peer-Reviewed Venues — `55` `dev`, `73` `held_out_test`, `45` `reserve` in Phase 3)**:
  - `6` curated `dev` papers: `P4-S3-ORCA-2022` (Yu et al., *Orca*, OSDI 2022), `P4-S3-PAGED-2023` (Kwon et al., *PagedAttention*, SOSP 2023), `P4-S3-SPECDEC-2023` (Leviathan, Kalman, & Matias, *Speculative Decoding*, ICML 2023), `P4-S3-SARATHI-2024` (Agrawal et al., *Sarathi-Serve*, OSDI 2024), `P4-S3-DISTSERVE-2024` (Zhong et al., *DistServe*, OSDI 2024), and `P4-S3-SGLANG-2024` (Zheng et al., *SGLang*, NeurIPS 2024), plus `49` `P4` checklist `dev` units from `mlsys-papers-2025-2026.yml` (`MLSys 2025`) and `systems-venues-2025.yml`.
  - `73` Phase 6 `held_out_test` units: `P4-S3-SLORA-2024` (Sheng et al., *S-LoRA*, MLSys 2024) plus `72` `P4` paper units from `systems-venues-2026.yml` (`OSDI`, `NSDI`, `EuroSys`, `ASPLOS 2026`, now burned to `dev`).
  - `45` Phase 6 `reserve` units from `mlsys-papers-2025-2026.yml` (`MLSys 2026` — unexposed to `Vocab-X`, already `held_out: false` in `0.4-draft`, now promoted to `dev`).
- **`S4` (Practitioner Artifacts & Diagnostic Scenarios — `2` `dev`, `36` `held_out_test`, `0` `reserve` in Phase 3)**:
  - `2` curated external `dev` guides: `P4-S4-ONNX-ORT-01` (`ONNX Runtime` / `Safetensors` / `Torch.export` packaging & execution providers) and `P4-S4-K8S-SERVING-01` (Kubernetes / `Ray Serve` / `Triton` rollouts, draining & backpressure).
  - `36` `StaffML` `P4` scenarios from `staffml-300-heldout-2026-10.yml` (`origin: internal`, Phase 6 `IDX`, now burned to `dev`).
- **`S5` (Standards & Benchmark Specifications — `2` `dev`, `0` `held_out_test`, `0` `reserve` in Phase 3)**:
  - `P4-S5-MLPERF-INF-01` (`dev`): Reddi et al., *MLPerf Inference Benchmark* (ISCA 2020) & MLPerf Inference v4.1 LoadGen scenario rules.
  - `P4-S5-MLPERF-TINY-01` (`dev`): Banbury et al., *MLPerf Tiny Benchmark* (NeurIPS Datasets & Benchmarks 2021).

### Area 2: `Q4` (`safety-assurance`) — Safety & assurance (`4` subjects, `30` `0.4-draft` concepts)
Subjects (`4`): `Q4.1` (`hazards-operating-domain`) Hazard analysis & operating domains · `Q4.2` (`runtime-enforcement`) Runtime assurance & safety filters · `Q4.3` (`verification-validation`) Safety verification & validation · `Q4.4` (`safety-cases`) Safety cases, standards & certification.

- **`S1` (Textbooks & Monographs — `46` `dev`, `2` `held_out_test`, `7` `reserve` in Phase 3)**:
  - `Q4-S1-LEVESON-STPA-01` (`dev`): Leveson, *Engineering a Safer World: Systems Thinking Applied to Safety* (MIT Press, 2011; foundational STPA monograph).
  - `Q4-S1-CONCRETE-SAFETY-01` (`dev`): Amodei et al., *Concrete Problems in AI Safety* (2016) & alignment monograph literature.
  - `44` `dev`, `2` `held_out_test` (`vol4/04_nervous`, excluded from Phase 6 `COV`), and `7` `reserve` (`vol4/16_release`) `MLSysBook` Vol. III–IV chapter units (`origin: internal`; all promoted to `dev` post-Phase 6).
- **`S2` (University Courses — `5` `dev`, `2` `held_out_test`, `1` `reserve` in Phase 3)**:
  - `Q4-S2-STANFORD-CS329T-01` (Stanford `CS329T` *Trustworthy Machine Learning*, `dev`) plus `4` `dev` checklist units from `courses-and-standards-2026.yml`; `Q4-S2-BERKELEY-CS294-267-01` (UC Berkeley `CS294-267` *AI Safety, Alignment, and Control*) plus `1` NIST checklist unit (`Q4-COV-courses-and-standards-2026-191`) in Phase 6 `held_out_test`; and `1` USC checklist unit (`Q4-COV-courses-and-standards-2026-208`) in Phase 6 `reserve` (all burned/`0.4`-exposed units promoted to `dev` post-Phase 6).
- **`S3` (Peer-Reviewed Venues — `5` `dev`, `1` `held_out_test`, `0` `reserve` in Phase 3)**:
  - `3` curated `dev` prose units: `Q4-S3-SIMPLEX-CBF-01` (Sha 2001 *Simplex Architecture* & Ames et al. 2019 *Control Barrier Functions*), `Q4-S3-NN-VERIFICATION-01` (Katz et al. CAV 2017 *Reluplex* & Liu et al. neural network verification survey), and `Q4-S3-RAND-DRIVING-01` (Kalra & Paddock RAND RR-1478 & Corso et al. *JAIR* 2021 black-box safety validation / adaptive stress testing survey), plus `2` `dev` venue-checklist units (`Q4-COV-mlsys-papers-2025-2026-028` [`MLSys 2025`] and `Q4-COV-systems-venues-2025-032` [`SOSP 2025`]).
  - `1` Phase 6 `held_out_test` unit: `Q4-COV-systems-venues-2026-186` (`ASPLOS 2026`, now burned to `dev`).
- **`S4` (Practitioner Artifacts & Diagnostic Scenarios — `1` `dev`, `4` `held_out_test`, `0` `reserve` in Phase 3)**:
  - `Q4-S4-FRONTIER-RSP-01` (`dev`): Frontier safety & deployment governance frameworks (Responsible Scaling / Preparedness / Frontier Safety capability thresholds & safeguard levels).
  - `4` `StaffML` `Q4` scenarios (`2` from `staffml-300-heldout-2026-10.yml` and `2` from `staffml-300-2026-10.yml`, `origin: internal`, Phase 6 `IDX`, now burned to `dev`).
- **`S5` (Standards & Assurance Guidance — `1` `dev`, `1` `held_out_test`, `1` `reserve` in Phase 3)**:
  - `Q4-S5-ISO26262-21448-UL4600` (`dev`): **ISO 26262:2018** (*Functional Safety*), **ISO 21448:2022** (*SOTIF*), and **UL 4600** / Goal Structuring Notation (`GSN`).
  - `Q4-S5-NIST-RMF-600-1` (Phase 6 `held_out_test`, now burned to `dev`): **NIST AI 100-1** (*AI RMF 1.0*) & **NIST AI 600-1** (*Generative AI Profile*, 2024).
  - `Q4-S5-ISO42001-PAS8800` (unexposed external `reserve`): **ISO/IEC 42001:2023** & **ISO/PAS 8800:2024** (*Road Vehicles — Safety and Artificial Intelligence*).
