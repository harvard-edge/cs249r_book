# Candidate External Editorial Board (Z39.19 §11.3.1.1)

Status: internal candidate roster for version `0.4-draft` (`2026-11-02`). **No candidate listed in this document has been contacted, invited, or asked to endorse the ML Systems Taxonomy.** Do not contact any candidate or publish external attributions without explicit authorization from the Editor (Vijay Janapa Reddi).

---

## 1. Purpose and Standard Grounding

Under **ANSI/NISO Z39.19-2005 (R2010) §11.1.3** (*Committee Approach*) and **§11.3.1.1** (*Review by an Editor and Board*), a controlled vocabulary intended for archival citation should combine empirical term extraction with review by domain subject experts and controlled-vocabulary specialists.

In `0.4-draft`, candidate discovery, triangulation, cross-checking, and publish-readiness auditing have been executed by three independent language-model families (`claude-opus-5-5`, `gpt-6-astra` / `gpt-6.1-sol`, `gemini-4-argon-high`) under deterministic reconciliation and LinkML compilation gates, with final decisions made by a single human Editor. To transition from an `editorial draft` to a citable community release (`1.0`), the project plans a seven-seat external review board:

- **Five Group Chairs (`F`, `S`, `P`, `Q`, `T`)**: one domain expert per top-level taxonomy group, responsible for reviewing subject boundaries, home-rule placements, and the 16 `specializes` edges where prior model audits split.
- **Two Controlled-Vocabulary & Knowledge-Organization Specialists (`V1`, `V2`)**: information science and thesaurus/SKOS specialists responsible for reviewing ANSI/NISO Z39.19 / ISO 25964-1 clause conformance, the human inter-indexer consistency protocol (`Hooper`, `Rolling`, $\alpha_{\text{MASI}}$), and SKOS/OWL crosswalks (`ACM CCS`, `CS2023`, `SFIA 9`, `ESCO`).

---

## 2. Candidate Roster (7 Seats, Primary + Alternate per Seat)

| Seat | Scope in Taxonomy | Primary Candidate | Alternate Candidate | Representative Work in Sampling Frame (`S1`–`S5`) | First-Pass Review Focus |
|---|---|---|---|---|---|
| **Group F** (`Foundations`) | `F1` Workloads & requirements · `F2` Models & architectures · `F3` Quantitative system models (`14` subjects) | **Christopher Ré** (Stanford University) | **Percy Liang** (Stanford University) | Stanford CS 229S / Dawn & Hazy Research data-centric & foundation-model systems; HELM (`S1`, `S2`) | Boundary between `F2` model architecture mechanisms (attention variants, state-space models) and `P4`/`S2` execution mechanisms; `F3` queueing and roofline scope notes |
| **Group S** (`Stack`) | `S1` Hardware & accelerators · `S2` Frameworks, compilers & runtimes · `S3` Parallelism & distributed execution · `S4` Datacenter infrastructure & fleet (`25` subjects) | **Tianqi Chen** (Carnegie Mellon University / OctoAI) | **Song Han** (MIT) | CMU 15-442 / 10-414 Deep Learning Systems (`S1`), Apache TVM / MLC-LLM (`S2`), MIT 6.5940 TinyML & Efficient Deep Learning (`S1`) | Compiler & runtime hierarchy (`S2.2`–`S2.4`), hardware-software co-design boundaries (`S1` vs. `S2`), and split `specializes` edges in collective communication (`S3.3`) |
| **Group P** (`Lifecycle`) | `P1` Data engineering · `P2` Training & adaptation · `P3` Model evaluation & testing · `P4` Deployment & serving · `P5` MLOps & operations (`25` subjects) | **Ion Stoica** (UC Berkeley) | **Matei Zaharia** (UC Berkeley) | UC Berkeley CS 294-267 (`S1`), Ray, vLLM (`PagedAttention`), Alpa, Chatbot Arena, compound AI systems (`S2`) | `P4` generative inference and compound AI serving (`P4.1`–`P4.9`), `P2` distributed RL post-training (`P2.4`), and promotion criteria for emerging serving architectures |
| **Group Q** (`Qualities`) | `Q1` Performance & benchmarking · `Q2` Reliability & fault tolerance · `Q3` Security & privacy · `Q4` Safety & assurance · `Q5` Responsible AI · `Q6` Sustainability & cost (`25` subjects) | **Sanjit A. Seshia** (UC Berkeley) | **Carole-Jean Wu** (Meta AI / Arizona State University) | Verified AI, formal methods and runtime assurance for cyber-physical ML (`Q4`); MLPerf and sustainable AI infrastructure carbon accounting (`Q1`, `Q6`, `S3`) | `Q4` safety cases, ODD, formal verification vs. runtime shielding (`Q4.1`–`Q4.4`), `Q1`/`Q6` metric-vs-lever separation, and alignment with `ISO/PAS 8800` and `UL 4600` (`S3`) |
| **Group T** (`System types`) | `T1` Agentic systems · `T2` Embodied & physical systems (`12` subjects) | **Luca Carlone** (MIT) | **Sergey Levine** (UC Berkeley) | MIT 16.485 Visual Navigation for Autonomous Vehicles (`S1`), certifiable perception and control in robotics (`T2`), embodied generalist policies | System-type admission test (`T1` and `T2` vs. general `P`/`Q` homes), timing & control-loop hierarchy (`T2.3`–`T2.4`), and `T1` agentic sandboxing vs. `Q3` access control |
| **Vocabulary & LIS (`V1`)** | ANSI/NISO Z39.19, ISO 25964-1/-2, faceted classification, and inter-indexer consistency trials | **Marcia Lei Zeng** (Kent State University) | **Claudio Gnoli** (University of Pavia / ISKO) | NISO Z39.19 / ISO 25964-1/-2 working group member; co-author of *Metadata* and *Knowledge Organization Systems* (`S3`) | Clause-level audit of `taxonomy/METHOD.md`, compound-term factoring rules (`Z39.19 §7`), facet-to-tree polyhierarchy (`Z39.19 §8.3.2`), and human indexer trial protocol (`§11.2.2`) |
| **SKOS / Semantic Web (`V2`)** | W3C SKOS Reference, OWL typed relation extensions, FAIR vocabulary publication, and external crosswalks | **Antoine Isaac** (Europeana / W3C SKOS Editor) | **Alistair Miles** (University of Oxford / W3C SKOS Editor) | Co-editor of the *W3C SKOS Reference* and *SKOS Primer*; ISO 25964-to-SKOS mapping specification (`S3`) | Validation of `mlsys-taxonomy.ttl` and `mlsys-taxonomy-relations.ttl` (17 typed OWL sub-properties of `skos:broader` and `skos:related`), `S27` disjointness, and `skos:exactMatch`/`skos:broadMatch` crosswalks |

---

## 3. Why Each Seat Fits the Z39.19 §11.3.1.1 Role

1. **Independence from single-vendor taxonomy lock-in**: Each domain candidate teaches a flagship university course in Stratum `S1` or leads open-source, multi-institution benchmarks and standards in Strata `S2`–`S3`.
2. **Direct coverage of the two-area empirical study (`P4` & `Q4`)**:
   - **Group P** (Stoica / Zaharia) directly covers the `P4` (`serving`) mechanisms evaluated in Phases 3–6 (`mlsys.paged-kv`, `mlsys.prefill-decode`, `mlsys.speculative-decoding`, `mlsys.compound-workflow-pipelining`, `mlsys.attention-ffn-disaggregation`).
   - **Group Q** (Seshia / Wu) directly covers the `Q4` (`safety-assurance`) mechanisms evaluated in Phases 3–6 (`mlsys.operational-design-domain`, `mlsys.simplex-architecture`, `mlsys.control-barrier-function`, `mlsys.neural-network-verification`, `mlsys.safety-case`, `mlsys.alarp`).
3. **Pairing domain engineering with formal knowledge organization (`V1`, `V2`)**: Z39.19 §11.1.3 notes that domain experts often over-split active research topics or conflate implementation tools with durable concepts, whereas vocabulary specialists enforce equivalence control, pre- vs. post-coordination, and hierarchical transitivity (`all-and-some` test).

---

## 4. Scoped Work Package for Board Review (When Authorized)

To keep external expert time bounded (estimated 90–120 minutes per reviewer), each board member receives a self-contained packet generated from the compiled release (`taxonomy/downloads/`):

1. **Group Subject & Concept Inventory** (`12`–`25` subjects, `90`–`240` concepts in their group): flag any missing `must`-have concept backed by $\ge 2$ provenance families, any product/benchmark masquerading as a concept (`Z39.19 §6.3.3`), or any misplaced home subject.
2. **Split Hierarchy Queue (`16` edges total across the taxonomy, ~`3` per group)**: decide each contested `specializes` edge using the 3-step `all-and-some` test (`Z39.19 §8.3.1`).
3. **Zero-Warrant Queue (`27` concepts total across the taxonomy, ~`5` per group)**: confirm external literary warrant (`S1`–`S4`) or recommend retirement/aliasing into a parent concept.
4. **Human Indexing Calibration Mini-Set (`15` items per group from `held_out_test`)**: assign 1 home subject and 1–3 concepts per item to establish the human inter-indexer baseline (`Hooper`, `Rolling`, $\alpha_{\text{MASI}}$, subject $\kappa$) required by `Z39.19 §11.2.2`.
