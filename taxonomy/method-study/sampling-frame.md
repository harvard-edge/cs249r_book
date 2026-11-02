# Enumerated Sampling Frame and Split Hygiene for the ML Systems Taxonomy

This document specifies the five-stratum sampling frame, provenance-family clustering rules, and split allocation protocol (`dev`, `held_out_test`, and `reserve`) synthesized from `Method-A` (§3), `Method-B` (§3), and audited `Method-C` (§3.2).

---

## 1. Five Stratified Source Populations (`S1`–`S5`)

Every candidate concept and every coverage evaluation unit must be drawn from an enumerated frame partitioned into five primary strata (`Method-A` §3.1, `Method-B` §3.1):

| Stratum ID | Stratum Name | Unit of Sampling | Inclusion Criteria & Warrant Role |
| :--- | :--- | :--- | :--- |
| **`S1`** | **Textbooks & Reference Monographs** | Chapter / major section | Published or open-access textbooks covering ML systems architecture, distributed training/serving, hardware acceleration, or safety/reliability engineering. Provides **backbone & curriculum warrant**. |
| **`S2`** | **University Courses** | Course syllabus (week/lecture + assignment/lab titles) | Graduate or upper-division undergraduate ML systems courses at accredited universities with public lecture schedules and reading lists, stratified across hardware/compilers, systems/networking, and safe/reliable deployment (`Method-A` §3.2). Provides **pedagogical warrant**. |
| **`S3`** | **Peer-Reviewed Systems & ML Venues** | Paper (title, abstract, section headings, system architecture section) | Papers from systems conferences (`MLSys`, `OSDI`, `SOSP`, `NSDI`, `ASPLOS`, `ISCA`, `EuroSys`) and systems/safety tracks of ML venues (`NeurIPS`, `ICML`, `ICLR`, `FAccT`, `SATML`) within a trailing 5-year window plus seminal classics (`>500` citations). Provides **literary warrant**. |
| **`S4`** | **Practitioner Artifacts, Postmortems & Role Archetypes** | Engineering design doc, incident postmortem, benchmark task, or role competency matrix | Public production engineering blogs/postmortems, open-source serving/runtime documentation (`vLLM`, `Triton`, `Ray Serve`, `ONNX Runtime` — used to extract underlying mechanisms, never product concepts), and senior/staff ML systems competency matrices (`StaffML`). Provides **user & operational warrant**. |
| **`S5`** | **Standards, Regulatory Frameworks & Benchmark Suites** | Standard clause / risk category / benchmark workload spec | Authoritative technical standards and assurance frameworks (`NIST AI 100-1`, `NIST AI 600-1`, `ISO/IEC 42001`, `ISO/IEC 23894`, `ISO/PAS 8800`, `IEEE 7000`, `MITRE ATLAS`, `OWASP Top 10 for LLM`, `MLPerf Inference/Training/Tiny`). Provides **normative & organizational warrant**. |

### Cross-Stratum Balance Quotas (`Method-C` §3.2, `Method-B` §3.2)
To prevent datacenter LLM serving from crowding out resource-constrained or classical ML systems (`Method-C` §3.2), the frame enforces explicit representation across execution scales where applicable to the area:
- **Datacenter / Cloud Scale** (multi-node, GPU/TPU cluster serving & training)
- **Enterprise / Single-Node / Workstation** (microservice orchestration, CPU/single-accelerator inference, RAG/compound pipelines)
- **Edge / On-Device / Embedded (`TinyML`)** (MCU/NPU/mobile quantization, memory planning, battery/thermal constraints)

---

## 2. Provenance-Family Clustering (`provenance_family_id`)

Before any random split or warrant count is performed, every source unit in the frame is assigned a `provenance_family_id` (`cluster_id`) (`Method-A` §2.1 & §3.3, `Method-B` §2.2 & §3.4):
1. **Shared Authorship / Lab Lineage**: Two sources whose primary/senior author sets overlap by $\ge 50\%$ (`[P]`) receive the same `provenance_family_id`.
2. **Derivative Educational & Code Artifacts**: A textbook chapter, a university course taught by the textbook's authors, slide decks derived from that textbook, and the companion question bank share a single `provenance_family_id` (`cluster_mlsysbook_harvard_staffml`).
3. **Paper–System–Benchmark Lineage**: A research paper, its open-source documentation, and a tutorial based on that paper share a single `provenance_family_id` (e.g., Kwon et al. SOSP 2023 PagedAttention paper + vLLM architecture docs = one cluster).

**Consequences of clustering**:
- Multiple items within the same `provenance_family_id` count as **at most one source** toward the two-source admission warrant rule.
- All items sharing a `provenance_family_id` are assigned to the **same split** (`dev`, `held_out_test`, or `reserve`) so that derivative artifacts never leak across discovery and evaluation splits.

---

## 3. Split Allocation Protocol (`dev`, `held_out_test`, `reserve`)

Synthesized from `Method-A` §3.3 and `Method-B` §3.4:
1. **Cluster-Stratified Random Assignment**:
   - Within each stratum (`S1`–`S5`) and test area (`P4`, `Q4`), provenance-family clusters are partitioned using a deterministic PRNG (`seed` pre-registered in `taxonomy/method-study/preregistration.yaml`):
     - **`dev` (Discovery & Construction Split, ~60%–70% of clusters)**: Accessible to builder agents during candidate harvesting and taxonomy construction.
     - **`held_out_test` (Blind Evaluation Split, ~20%–25% of clusters)**: Withheld from builder agents; used only by read-only evaluation judges in Phase 6 to measure topic coverage and multi-rater indexing agreement.
     - **`reserve` (Sequestered Reserve Split, ~10%–15% of clusters)**: Held in reserve to replace any `held_out_test` cluster that becomes burned after diagnostic inspection (`Method-B` §3.4).
2. **Treatment of Project-Internal Corpora (`MLSysBook` Chapters & `StaffML` Questions)**:
   - Because the existing taxonomy (`0.4-draft`) was co-developed with the `MLSysBook` manuscript and `StaffML` questions, external courses (`S2`), external papers (`S3`), and external standards (`S5`) provide the primary uncontaminated held-out test signal when comparing a rebuilt vocabulary against the published vocabulary (`Method-B` §2.1).
   - For the Phase 4–6 two-area test (`P4` and `Q4`), the pre-registered frame includes both external clusters (`S2`, `S3`, `S4`, `S5`) and held-out book/StaffML units so both external generalizability and internal application fit are measured and reported separately.

---

## 4. Candidate Sampling Frame for the Two Test Areas (`P4` and `Q4`)

In Phase 3 (immediately after Decision Gate 1 approval), the exact document manifest and PRNG draw will be committed in `taxonomy/method-study/preregistration.yaml`. The candidate source pool for the two test areas is:

### Area 1: `P4` — Deployment & Serving
Current subjects (`6`): `P4.1` Model packaging & export, `P4.2` Serving architectures, `P4.3` Generative & LLM inference, `P4.4` Compound AI & routing, `P4.5` On-device & edge deployment, `P4.6` Capacity planning & autoscaling.

- **`S1` (Textbooks)**:
  - `MLSysBook` Serving, Edge/Mobile Deployment, and Fleet Operations chapters (`dev` + held-out sections).
  - Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022) — Ch. 4 (Training Data / Feature Stores), Ch. 7 (Model Deployment & Prediction Service), Ch. 9 (Continual Learning & Testing in Production).
- **`S2` (University Courses)**:
  - CMU `15-442 / 15-642` (*Machine Learning Systems*) — Serving, Compilation, and Edge/Mobile modules.
  - Stanford `CS229S` (*Systems for Machine Learning*) — Inference serving, quantization, speculative decoding, and LLM runtime lectures.
  - MIT `6.5940` (*TinyML and Efficient Deep Learning Computing*, Han) — On-device inference, pruning, quantization, KV-cache compression, and mobile/MCU deployment lectures.
  - UC Berkeley `CS294-252 / CS294-194` (*Large Language Model Systems / Compound AI Systems*) — Serving engines, disaggregated prefill/decode, routing, and agent/tool orchestration lectures.
- **`S3` (Peer-Reviewed Systems Venues)**:
  - Crankshaw et al., *Clipper: A Low-Latency Online Prediction Serving System* (NSDI 2017).
  - Gujarati et al., *Serving DNNs like Clockwork: Performance Predictability from the Bottom Up* (OSDI 2020).
  - Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022).
  - Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023).
  - Leviathan et al. / Chen et al., *Fast Inference from Transformers via Speculative Decoding* (ICML 2023).
  - Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (OSDI 2024).
  - Zhong et al., *DistServe: Disaggregating Prefill and Decode for Goodput-Optimized Large Language Model Serving* (OSDI 2024).
  - Sheng et al., *S-LoRA: Serving Thousands of Concurrent LoRA Adapters* (MLSys 2024).
  -Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (NeurIPS 2024).
  - Lin et al., *MCUNet: Tiny Deep Learning on IoT Devices* (NeurIPS 2020) / *AWQ: Activation-aware Weight Quantization* (MLSys 2024).
- **`S4` (Practitioner Artifacts & Role Archetypes)**:
  - Open-source architecture specifications for model packaging/interchange (`ONNX` / `Safetensors` / `TorchScript` export boundary specs), Kubernetes/Ray Serve autoscaling & backpressure design docs, and `StaffML` `P4` diagnostic questions.
- **`S5` (Standards & Benchmark Specifications)**:
  - Reddi et al., *MLPerf Inference Benchmark* (ISCA 2020) & MLPerf Inference v4.x server/offline/edge scenario rules (TTFT, TPOT, latency-bounded throughput constraints).
  - Banbury et al., *MLPerf Tiny Benchmark* (NeurIPS Datasets & Benchmarks 2021).

### Area 2: `Q4` — Safety & Assurance
Current subjects (`4`): `Q4.1` Hazard analysis & assurance cases, `Q4.2` Alignment & runtime guardrails, `Q4.3` Verification & formal bounds, `Q4.4` Pre-deployment safety testing & red-teaming.

- **`S1` (Textbooks & Monographs)**:
  - `MLSysBook` Safety, Security, Responsible AI, and Assurance chapters (`dev` + held-out sections).
  - Leveson, *Engineering a Safer World: Systems Thinking Applied to Safety* (MIT Press, 2011) — STPA hazard analysis and safety control structures as applied to autonomous/ML systems.
- **`S2` (University Courses)**:
  - Stanford `CS329T` (*Trustworthy Machine Learning*) — Robustness, formal verification, red-teaming, watermarking, and guardrail modules.
  - UC Berkeley `CS294-267` (*AI Safety and Alignment*) — Threat modeling, capability evaluations, sandboxing, and runtime control modules.
  - MIT `6.S986` / Safe & Reliable ML modules — Neural network verification, conformal/certified bounds, and safety cases.
- **`S3` (Peer-Reviewed Venues)**:
  - Katz et al., *Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks* (CAV 2017) / Wang et al., *Beta-CROWN* (NeurIPS 2021).
  - Cohen et al., *Certified Adversarial Robustness via Randomized Smoothing* (ICML 2019).
  - Perez et al., *Red Teaming Language Models with Language Models* (EMNLP 2022) / Ganguli et al. (2022).
  - Greshake et al., *Indirect Prompt Injection in LLM-Integrated Applications* (AISec 2023).
  - Greenblatt et al., *AI Control: Improving Safety Despite Intentional Subversion* (ICML 2024).
  - Hawkins et al., *Guidance on the Assurance of Machine Learning in Autonomous Systems (AMLAS)* (University of York / Assuring Autonomy International Programme, 2021) — GSN assurance cases for ML.
- **`S4` (Practitioner Artifacts & Role Archetypes)**:
  - Frontier safety & deployment frameworks (Anthropic Responsible Scaling Policy / OpenAI Preparedness Framework / Google DeepMind Frontier Safety Framework capability thresholds & deployment mitigations), plus `StaffML` `Q4` diagnostic questions.
- **`S5` (Standards & Regulatory Frameworks)**:
  - **NIST AI 100-1** (*AI Risk Management Framework 1.0*: Govern, Map, Measure, Manage) & **NIST AI 600-1** (*Generative AI Profile*, 2024).
  - **ISO/IEC 42001:2023** (*Artificial Intelligence Management System*) & **ISO/PAS 8800:2024** (*Road Vehicles — Safety and Artificial Intelligence*).
  - **OWASP Top 10 for Large Language Model Applications** (v1.1 / v2.0: prompt injection, insecure output handling, excessive agency, model DoS, supply chain vulnerabilities).
  - **MITRE ATLAS** (*Adversarial Threat Landscape for Artificial-Intelligence Systems*).
