# Enumerated Sampling Frame and Split Hygiene for the ML Systems Taxonomy

This document specifies the five-stratum sampling frame, provenance-family clustering rules, and split allocation protocol (`dev`, `held_out_test`, and `reserve`) synthesized from `Method-A` (§3), `Method-B` (§3), and audited `Method-C` (§3.2), together with the executed Phase 3 allocation for `P4` and `Q4` (`taxonomy/method-study/preregistration.yaml`).

*Note: Sampling-frame strata `S1`–`S5` denote source populations and are distinct from Stack area codes `S1`–`S4`.*

---

## 1. Five Stratified Source Populations (`S1`–`S5`)

Every candidate concept and every coverage or indexing evaluation unit is drawn from an enumerated frame partitioned into five source populations (`Method-A` §3.1, `Method-B` §3.1). Source stratum records **where an excerpt comes from**; warrant type under ANSI/NISO Z39.19 §5.3.5.1–§5.3.5.3 is recorded separately so operational proxies are not conflated with direct user search logs or organizational term-preference mandates:

| Stratum ID | Stratum Name | Unit of Sampling | Inclusion Criteria & Warrant Role |
| :--- | :--- | :--- | :--- |
| **`S1`** | **Textbooks & Reference Monographs** | Chapter / major section | Published or open-access textbooks covering ML systems architecture, distributed training/serving, hardware acceleration, or safety/reliability engineering (both external monographs and project-internal `MLSysBook` chapters). Provides **literary & curriculum warrant** (`Z39.19 §5.3.5.1`; internal `MLSysBook` chapters carry `origin: internal`). |
| **`S2`** | **University Courses** | Course syllabus (lecture module + assignment/lab scope) | Graduate or upper-division undergraduate ML systems courses at accredited universities with public lecture schedules and reading lists, stratified across hardware/compilers, systems/networking, and safe/reliable deployment (`Method-A` §3.2). Provides **external literary & pedagogical warrant**. |
| **`S3`** | **Peer-Reviewed Systems & ML Venues** | Paper (title, abstract, architecture/evaluation passage, or blind venue topic entry) | Peer-reviewed papers from systems conferences (`MLSys`, `OSDI`, `SOSP`, `NSDI`, `ASPLOS`, `ISCA`, `EuroSys`, `CAV`, `AISec`) and systems/safety tracks of ML/NLP venues (`NeurIPS`, `ICML`, `ICLR`, `EMNLP`, `FAccT`, `SATML`) within a trailing 5-year window (`2021`–`2026` at study freeze) plus pre-registered foundational papers (`>500` citations in Google Scholar / OpenAlex or standard `S2` syllabus adoption). Provides **external literary warrant**. |
| **`S4`** | **Practitioner Artifacts, Postmortems & Diagnostic Scenarios** | Engineering design doc, incident postmortem, runtime spec, or diagnostic scenario | Public production engineering design guides, open-source serving/runtime documentation (`Triton`, `Ray Serve`, `ONNX Runtime` — used solely to extract underlying mechanisms, never product concepts), frontier deployment governance policies, and `StaffML` diagnostic scenarios (`origin: internal`). Provides **operational literary warrant** and (for `StaffML` tag counts) an **assignment-frequency proxy**, not direct user search/request warrant (`Z39.19 §5.3.5.3`). |
| **`S5`** | **Standards, Assurance Guidance & Benchmark Specifications** | Standard clause / assurance guidance module / benchmark scenario specification | Published technical standards, public assurance guidance reports, and benchmark specifications (`NIST AI 100-1`, `NIST AI 600-1`, `ISO 26262`, `ISO 21448`, `ISO/IEC 42001`, `ISO/PAS 8800`, `UL 4600`, `AMLAS`, `OWASP Top 10 for LLM`, `MITRE ATLAS`, `MLPerf Inference/Training/Tiny`). Provides **normative literary warrant** (`Z39.19 §5.3.5.1`), not internal organizational preferred-term mandates (`Z39.19 §5.3.5.2`) unless term-form adoption is separately logged. |

### Cross-Stratum Scale Representation (`Method-C` §3.2, `Method-B` §3.2)
To prevent datacenter LLM serving from crowding out resource-constrained or classical ML systems, the frame requires representation across execution scales where applicable to an area:
- **Datacenter / Cloud Scale** (multi-node accelerator cluster serving, training, and fleet scheduling)
- **Enterprise / Single-Node / Workstation** (gateway/microservice orchestration, CPU/single-accelerator inference, retrieval and compound pipelines)
- **Edge / On-Device / Embedded (`TinyML`)** (MCU/NPU/mobile quantization, static arena memory planning, battery/thermal envelopes)

---

## 2. Global Provenance-Family Clustering (`provenance_family_id`)

Before stratification, split assignment, or warrant counting, every unit in `<campaign>/sampling-manifest.json` receives an explicit `unit_id`, `provenance_family_id`, `stratum` (`S1`–`S5`), `split` (`dev`, `held_out_test`, `reserve`), `origin` (`external` | `internal`), and `source_type` (`prose` | `heading`) (`Method-A` §2.1 & §3.3, `Method-B` §2.2 & §3.4):

1. **Shared Authorship / Lab Lineage (Deterministic Overlap & Transitive Closure)**: Let $A_i$ denote the set of first and last (primary and senior) authors of source $i$. Two sources $i, j$ are linked when $|A_i \cap A_j| / \min(|A_i|, |A_j|) \ge 0.50$, and connected components (transitive closure) share a single global `provenance_family_id` across all strata and areas.
2. **Paper–System–Benchmark Lineage**: A research paper, its companion open-source documentation, and tutorials derived from it share one `provenance_family_id` (for example, Kwon et al. SOSP 2023 PagedAttention + vLLM architecture docs = one cluster).
3. **Project-Internal Corpora (`origin: internal`)**: All units drawn from `MLSysBook` chapters (`pf_mlsysbook_*`), `Harvard CS249r` course materials (`pf_harvard_cs249r_*`), and `StaffML` diagnostic banks (`pf_staffml_*`) carry `origin: internal` (`INTERNAL_LINEAGE_PREFIXES` in `reconcile.py`):
   - **Gate 2 external-warrant rule**: Any number of `origin: internal` units supplies `0` external lineages; every new concept in `concepts_add` requires $\ge 2$ distinct `provenance_family_id` clusters including $\ge 1$ `origin: external` cluster and $\ge 1$ `source_type: prose` unit.
   - **Chapter/pool split hygiene**: In the executed Phase 3 manifest (`preregistration.yaml`), `MLSysBook` units were grouped by chapter (`pf_mlsysbook_<vol>/<chapter>`) and `StaffML` units by sample pool (`pf_staffml_dev_*`, `pf_staffml_heldout_*`), with `0` `provenance_family_id` clusters crossing `dev`, `held_out_test`, or `reserve`. Because chapters and `StaffML` questions share project authorship (`0.4-draft` was co-developed alongside `MLSysBook` and `StaffML` Sample 1), `origin: internal` units are **isolated from external generalization claims in evaluation**: all `7` held-out `MLSysBook` chapter units were excluded from Phase 6 coverage (`COV` used `79` `origin: external` `S2`/`S3`/`S5` units only), and Phase 6 indexing (`IDX`) is reported separately for `origin: internal` (`40` `StaffML` units, including the `38` `staffml-300-heldout-2026-10` units never used to tune `0.4-draft`) versus `origin: external` (`3` `Q4` `S2`/`S5` units). For prospective refreshes across new areas, grouping all project-internal derivatives into `dev` (or evaluating held-out generalizability exclusively on `origin: external` clusters) prevents internal cross-artifact leakage.

**Mechanical enforcement (`reconcile.py`)**:
- `reconcile.py` requires `<campaign>/sampling-manifest.json` whenever `concepts_add` or `upward-posting` aliases appear, rejects duplicate `unit_id`s or missing `split`/`origin`/`source_type` fields, and fails closed if any `provenance_family_id` appears in more than one split (`dev`, `held_out_test`, `reserve`), even when a cluster spans multiple areas (such as `pf_cov_sysvenues_2026_heldout_asplos2026` spanning `P4` and `Q4` inside `held_out_test`).

---

## 3. Split Allocation Protocol (`dev`, `held_out_test`, `reserve`)

1. **Prospective Refresh Targets (`60%` / `25%` / `15%`)**:
   - In periodic refreshes (`.claude/workflows/taxonomy-refresh.md`, `taxonomy/METHOD.md`), global `provenance_family_id` clusters are assigned once before construction with target cluster proportions of **`60%` `dev`** (visible to builders), **`25%` `held_out_test`** (sealed until Step 5b evaluation), and **`15%` `reserve`** (sequestered to replenish burned `held_out_test` clusters alongside fresh post-cutoff external sources). Multi-area clusters are assigned as whole clusters prior to per-area tallies so a cluster never splits across partitions.
2. **Executed Phase 3 Allocation in the Two-Area Study (`preregistration.yaml`, `seed = 20261103`)**:
   - In the completed Phase 3 `P4`+`Q4` study (`382` units across `97` unique `provenance_family_id` clusters; `0` cross-split clusters), `build_phase3_preregistration.py` combined pre-designated benchmark splits (`fixed_split`, assigning curated single-passage `S1`–`S5` study excerpts to `dev`, the `2026` systems-venue checklists and `staffml-300-heldout-2026-10` pool plus `NIST AI RMF` to `held_out_test`, and `ISO/IEC 42001 / ISO/PAS 8800` to `reserve`) with a deterministic cell-wise draw (`seed = 20261103`) over movable `(area, stratum)` cells of size $n$ (`1` cluster to `held_out_test` when $n \ge 3$, `1` cluster to `reserve` when $n \ge 4$, and $n - 2$ clusters to `dev`).
   - Because single-passage curated sources each formed a 1-unit `dev` cluster while `held_out_test` and `reserve` held larger venue/question-bank clusters (`pf_cov_sysvenues_2026_heldout_*`, `pf_staffml_heldout_*`, `pf_cov_mlsys_25_26_mlsys2026`), the **achieved cluster split departed upward in `dev` (`76 / 97 = 78.4%` vs. the `60%` prospective target)** while the **achieved unit split was `200 / 382` (`52.4%`) `dev`, `126 / 382` (`33.0%`) `held_out_test`, and `56 / 382` (`14.7%`) `reserve`**:
     - **Clusters by split (`97` total)**: `76` `dev` (`78.4%`: `47` `P4`-only, `23` `Q4`-only, `6` `P4`/`Q4`-shared; `45` `external`, `31` `internal`), `15` `held_out_test` (`15.5%`: `8` `P4`-only, `6` `Q4`-only, `1` `P4`/`Q4`-shared; `10` `external`, `5` `internal`), and `6` `reserve` (`6.2%`: `3` `P4`-only, `3` `Q4`-only; `4` `external`, `2` `internal`). *(Note: `preregistration.yaml` records `area` by first unit occurrence, tabulating the `6` shared `dev` clusters under `P4` as `53` `P4` + `23` `Q4` = `76`.)*
     - **Units by split, area, and origin (`382` total)**: `dev` = `200` (`142` `P4` [`95` `external`, `47` `internal`], `58` `Q4` [`14` `external`, `44` `internal`]); `held_out_test` = `126` (`116` `P4` [`75` `external`, `41` `internal`], `10` `Q4` [`4` `external`, `6` `internal`]); `reserve` = `56` (`47` `P4` [`46` `external`, `1` `internal`], `9` `Q4` [`2` `external`, `7` `internal`]).
     - Because `126` `held_out_test` units span `15` clusters (`8.4` units/cluster vs. `2.6` in `dev`), Phase 6 uncertainty in `METHOD.md` reports `10,000`-resample **cluster-level bootstrap confidence intervals** alongside unit-level intervals, and future `P4`/`Q4` evaluations must ingest fresh post-cutoff external clusters because the `56`-unit `reserve` pool (`47` `P4`, `9` `Q4`) cannot alone replace the `126` burned `held_out_test` units.

---

## 4. Executed Two-Area Sampling Frame (`P4` and `Q4`)

In Phase 3 (`2026-11-02`), the exact `382`-unit manifest and deterministic split (`random_seed: 20261103`) were committed in `taxonomy/method-study/preregistration.yaml` (enumerating `S1`–`S5` source pools for the remaining `18` taxonomy areas remains open pre-`1.0` work). The two study areas and their canonical `0.4-draft` subjects (`taxonomy/ml-systems-taxonomy.yml`) are:

### Area 1: `P4` (`serving`) — Deployment & serving (`9` subjects, `95` `0.4-draft` concepts)
Subjects (`9`): `P4.1` (`serving-architecture`) Serving architectures · `P4.2` (`compound-ai`) Compound AI systems & model routing · `P4.3` (`latency-goodput`) Latency, throughput & goodput · `P4.4` (`request-scheduling`) Batching, admission & autoscaling · `P4.5` (`llm-inference`) LLM & generative inference · `P4.6` (`structured-generation`) Decoding, sampling & structured output · `P4.7` (`test-time-compute`) Test-time compute · `P4.8` (`retrieval-rag`) Retrieval & vector search · `P4.9` (`on-device-deployment`) On-device & edge deployment.

- **`S1` (Textbooks & Monographs — `50` `dev`, `5` `held_out_test`, `1` `reserve` units)**:
  - Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022) — Ch. 7–8 (batch vs. online prediction, dynamic batching, export/compression).
  - Barroso, Hölzle, & Ranganathan, *The Datacenter as a Computer* (3rd ed., Morgan & Claypool, 2018; pre-2021 foundational monograph) — Ch. 2 & 6 (tail latency, hedging, admission control).
  - `MLSysBook` Volumes I–IV primary chapter topics (`origin: internal`; `5` held-out `vol1/13_model_serving` units excluded from Phase 6 `COV`).
- **`S2` (University Courses — `33` `dev`, `2` `held_out_test`, `1` `reserve` units)**:
  - CMU `11-868` (*Large Language Model Systems*, Spring 2025), MIT `6.5940` (*TinyML and Efficient Deep Learning Computing*, Fall 2024), Stanford `CS336` (*Language Modeling from Scratch*, 2025, `held_out_test`), UT Austin `CS378` (`reserve`), plus `P4`-scoped topics from `taxonomy/coverage/courses-and-standards-2026.yml`.
- **`S3` (Peer-Reviewed Venues — `55` `dev`, `73` `held_out_test`, `45` `reserve` units)**:
  - Pre-2021 foundational systems papers (`>500` Scholar citations / `S2` syllabus staples): Crankshaw et al., *Clipper: A Low-Latency Online Prediction Serving System* (NSDI 2017); Gujarati et al., *Serving DNNs like Clockwork: Performance Predictability from the Bottom Up* (OSDI 2020); Lin et al., *MCUNet: Tiny Deep Learning on IoT Devices* (NeurIPS 2020).
  - 2021–2026 peer-reviewed papers: Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022); Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023); Leviathan, Kalman, & Matias, *Fast Inference from Transformers via Speculative Decoding* (ICML 2023) [distinct from the concurrent preprint Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, arXiv:2302.01318, 2023]; Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (OSDI 2024); Zhong et al., *DistServe: Disaggregating Prefill and Decode for Goodput-Optimized Large Language Model Serving* (OSDI 2024); Sheng et al., *S-LoRA: Serving Thousands of Concurrent LoRA Adapters* (MLSys 2024, `held_out_test`); Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (NeurIPS 2024); Lin et al., *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration* (MLSys 2024); plus `P4` checklists in `mlsys-papers-2025-2026.yml`, `systems-venues-2025.yml`, and blind `systems-venues-2026.yml` (`72` `P4` units in `held_out_test`).
- **`S4` (Practitioner Artifacts & Diagnostic Scenarios — `2` `dev`, `36` `held_out_test`, `0` `reserve` units)**:
  - `ONNX Runtime` / `Safetensors` / `torch.export` packaging architecture guides, Kubernetes / `Ray Serve` / `Triton` gateway & backpressure guides, and `36` held-out `StaffML` `P4` scenarios (`staffml-300-heldout-2026-10.yml`, `origin: internal`).
- **`S5` (Standards & Benchmark Specifications — `2` `dev`, `0` `held_out_test`, `0` `reserve` units)**:
  - Reddi et al., *MLPerf Inference Benchmark* (ISCA 2020; foundational benchmark spec) & MLPerf Inference v4.1 scenario rules; Banbury et al., *MLPerf Tiny Benchmark* (NeurIPS Datasets & Benchmarks 2021).

### Area 2: `Q4` (`safety-assurance`) — Safety & assurance (`4` subjects, `30` `0.4-draft` concepts)
Subjects (`4`): `Q4.1` (`hazards-operating-domain`) Hazard analysis & operating domains · `Q4.2` (`runtime-enforcement`) Runtime assurance & safety filters · `Q4.3` (`verification-validation`) Safety verification & validation · `Q4.4` (`safety-cases`) Safety cases, standards & certification.

- **`S1` (Textbooks & Monographs — `46` `dev`, `2` `held_out_test`, `7` `reserve` units)**:
  - Leveson, *Engineering a Safer World: Systems Thinking Applied to Safety* (MIT Press, 2011; foundational STPA monograph).
  - `MLSysBook` Volumes III–IV safety and assurance chapter topics (`origin: internal`; `2` held-out `vol4/04_nervous` units excluded from Phase 6 `COV`).
- **`S2` (University Courses — `5` `dev`, `2` `held_out_test`, `1` `reserve` units)**:
  - Stanford `CS329T` (*Trustworthy Machine Learning*, `dev`), UC Berkeley `CS294-267` (*Safety and Control for AI*, `held_out_test`), plus `Q4`-scoped topics from `courses-and-standards-2026.yml`.
- **`S3` (Peer-Reviewed Venues — `5` `dev`, `1` `held_out_test`, `0` `reserve` units)**:
  - Pre-2021 foundational verification papers (`>500` Scholar citations): Katz et al., *Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks* (CAV 2017); Cohen, Rosenfeld, & Kolter, *Certified Adversarial Robustness via Randomized Smoothing* (ICML 2019).
  - 2021–2026 peer-reviewed papers & surveys: Wang et al., *Beta-CROWN* (NeurIPS 2021); Corso et al., survey of black-box safety validation and adaptive stress testing (*JAIR* 2021, paired with RAND RR-1478 statistical exposure framing); Perez et al., *Red Teaming Language Models with Language Models* (EMNLP 2022); Greshake et al. (AISec 2023, indirect prompt injection in LLM-integrated applications, cross-area with `Q3`); Greenblatt et al., *AI Control: Improving Safety Despite Intentional Subversion* (ICML 2024); plus `Q4` entries in `mlsys-papers-2025-2026.yml`, `systems-venues-2025.yml`, and `systems-venues-2026.yml` (`1` `ASPLOS 2026` unit in `held_out_test`).
- **`S4` (Practitioner Artifacts & Diagnostic Scenarios — `1` `dev`, `4` `held_out_test`, `0` `reserve` units)**:
  - Frontier safety & deployment governance frameworks (Responsible Scaling / Preparedness / Frontier Safety capability thresholds & safeguard levels, `dev`), plus `4` held-out `StaffML` `Q4` scenarios (`2` from `staffml-300-heldout-2026-10.yml` and `2` from `staffml-300-2026-10.yml`, `origin: internal`).
- **`S5` (Standards & Assurance Guidance Frameworks — `1` `dev`, `1` `held_out_test`, `1` `reserve` units)**:
  - **ISO 26262:2018** (*Functional Safety*), **ISO 21448:2022** (*SOTIF*), **UL 4600**, and **Hawkins et al. (2021) *AMLAS* guidance report** (University of York Assuring Autonomy International Programme technical assurance guidance, classified in `S5` rather than peer-reviewed `S3`; `dev`).
  - **NIST AI 100-1** (*AI RMF 1.0*) & **NIST AI 600-1** (*Generative AI Profile*, 2024) (`held_out_test`).
  - **ISO/IEC 42001:2023** & **ISO/PAS 8800:2024** (*Road Vehicles — Safety and Artificial Intelligence*) (`reserve`), alongside security-adjacent `OWASP Top 10 for LLM` and `MITRE ATLAS` boundaries (`Q3`).
