# External Editorial Board Charter and Human Evaluation Protocol (Z39.19 §11.1.3.1, §11.2.2, §11.3.1.1)

Status: public governance charter and role specification for version `0.4-draft` (`2026-11-02`). **This document defines seat qualifications, area coverage obligations, conflict-of-interest rules, and human evaluation packets only; no external board has yet been convened, no individual has been contacted or asked to endorse the ML Systems Taxonomy, and unconsented personal names are intentionally excluded from the public repository.**

---

## 1. Purpose and Historical Context

Under **ANSI/NISO Z39.19-2005 (R2010) §11.1.3.1** (*Committee Approach*), **§11.2.2** (*Evaluation Criteria*), and **§11.3.1.1** (*Addition of Terms*), a controlled vocabulary intended for community reuse combines empirical corpus extraction with review by domain subject specialists and controlled-vocabulary specialists.

In `0.4-draft`, construction and verification used language-model passes under deterministic reconciliation and LinkML compilation gates, with final decisions made by a single human Editor (Vijay Janapa Reddi): specifically, the `0.4-draft` refresh ran Round 1 as seven parallel lenses in one model family (`claude-fable-5-1`) followed by a three-family Round 2 cross-check and independent audit (`gpt-6.1-sol`, `gemini-4-argon-high`, `claude-opus-5-5`), while full three-family blind discovery and reconciliation were evaluated on the pre-registered Phase 3–7 two-area study (`P4` and `Q4`) and mandated for future refreshes.

To transition from an `editorial draft` (`0.x-draft`) to a citable community release (`1.0`), the project requires review by a **7-seat external editorial board** backed by mandatory sub-area specialist reviewers covering all 20 taxonomy areas and a **12-indexer dual-human inter-indexer consistency study**.

---

## 2. Board Structure, Voting Seats, and Full 20-Area Specialist Coverage

*Note: Sampling-frame strata `S1`–`S5` (`S1` textbooks/monographs, `S2` university courses, `S3` peer-reviewed systems/ML venues, `S4` practitioner artifacts and StaffML questions, `S5` standards and regulatory frameworks) are distinct from Stack area codes `S1`–`S4`.*

The board comprises **7 voting seats** (`5` domain group seats + `2` vocabulary/SKOS seats). Because several taxonomy groups span distinct engineering sub-disciplines, convening a release review requires both the primary/alternate seat holders and the named specialist reviewers below so that **all 20 areas (`101` subjects)** have explicit domain coverage:

| Voting Seat | Taxonomy Scope | Seat Composition (Participating Reviewers) | Required Qualifications & Stratum Grounding | Mandatory Sub-Area Specialist Coverage (Prerequisite Before Release Sign-Off) |
|---|---|---|---|---|
| **Seat `F`** (`Foundations`, 1 vote) | `F1` Workloads & requirements · `F2` Models & architectures · `F3` Quantitative system models (`14` subjects) | 1 Chair + 1 Alternate from distinct institutions, plus 1 `F3` specialist reviewer | Published research and graduate teaching (`S1`–`S3`) in ML workloads, foundation-model architectures, and analytical performance/scaling models. | Dedicated **queueing, roofline, and analytical performance-modeling specialist** required for `F3` alongside `F1`/`F2` architecture coverage. |
| **Seat `S`** (`Stack`, 1 vote) | `S1` Hardware & accelerators · `S2` Frameworks, compilers & runtimes · `S3` Parallelism & distributed execution · `S4` Datacenter infrastructure & fleet (`25` subjects) | 1 Chair + 1 Alternate from distinct institutions, plus 1 `S4` specialist reviewer | Graduate systems courses (`S2`) and peer-reviewed compiler, accelerator, or distributed-execution systems (`S3`). | Dedicated **datacenter networking, cluster scheduling, and power/thermal fleet infrastructure specialist** required for `S4` alongside `S1`–`S3` hardware/compiler/parallelism coverage. |
| **Seat `P`** (`Lifecycle`, 1 vote) | `P1` Data engineering · `P2` Training & adaptation · `P3` Model evaluation & testing · `P4` Deployment & serving · `P5` MLOps & operations (`25` subjects) | 1 Chair + 1 Alternate from distinct institutions | Peer-reviewed serving, training, data-pipeline, and evaluation systems (`S3`) plus production deployment artifacts (`S4`). | Must jointly cover `P1`/`P5` data & MLOps pipelines, `P2` distributed pretraining/RL post-training, `P3` evaluation protocols, and `P4` serving (`P4.1`–`P4.9`). |
| **Seat `Q`** (`Qualities`, 1 vote) | `Q1` Performance, efficiency & benchmarking · `Q2` Reliability & fault tolerance · `Q3` Security & privacy · `Q4` Safety & assurance · `Q5` Responsible AI · `Q6` Sustainability & cost (`25` subjects) | 1 Chair + 1 Alternate from distinct institutions, plus 4 targeted area specialists | Standards participation (`S5`) and peer-reviewed research (`S3`) across system benchmarking/efficiency, formal/runtime assurance, security/privacy, and governance. | Because no two individuals span all six quality areas, release review requires explicit specialist coverage for: (i) `Q1`/`Q6` benchmarking & carbon/cost accounting, (ii) `Q2` fault-tolerant distributed systems, (iii) `Q3` ML security & differential privacy, (iv) `Q4` **both** formal/runtime verification **and** safety cases / system-theoretic hazard analysis (`STPA`, `FMEA`, `FTA`, `UL 4600`, `SOTIF`), and (v) `Q5` fairness & AI governance. |
| **Seat `T`** (`System types`, 1 shared vote) | `T1` Agentic systems · `T2` Embodied & physical systems (`12` subjects) | **2 mandatory Joint Co-Chairs** (`1` for `T1` Agentic, `1` for `T2` Embodied) from distinct institutions | `T1`: agentic harnesses, tool-use security, and multi-agent evaluation (`S2`–`S4`). `T2`: robotics, autonomous cyber-physical control, real-time sensing, and WCET (`S2`, `S3`, `S5`). | Both Co-Chairs must review and concur on the Group `T` vote (if they split on a `T1` or `T2` item, the item escalates to full-board adjudication with `V1`). |
| **Seat `V1`** (`Controlled Vocabulary & LIS`, 1 vote) | ANSI/NISO Z39.19, ISO 25964-1/-2, faceted classification, equivalence/compound rules, and human indexer study | 1 Chair + 1 Alternate | Scholarship or standards-committee experience in knowledge organization systems (`KOS`), thesaurus construction (`Z39.19` / `ISO 25964`), and inter-indexer consistency evaluation. | Affirmative sign-off from `V1` is required on `taxonomy/METHOD.md` clause conformance and the pre-registered human inter-indexer protocol before `1.0` release. |
| **Seat `V2`** (`SKOS / Semantic Web`, 1 vote) | W3C SKOS Reference, OWL typed relation extensions, FAIR vocabulary publication, and external crosswalks | 1 Chair + 1 Alternate | Authorship or editorial maintenance of published W3C SKOS/OWL vocabularies, RDF validation pipelines, and ISO 25964-to-SKOS mappings. | Affirmative sign-off from `V2` is required on `mlsys-taxonomy.ttl`, `mlsys-taxonomy-relations.ttl`, `mlsys-taxonomy.dcat.ttl`, and external SKOS crosswalks before `1.0` release. |

---

## 3. Governance Safeguards, Quorum, and Conflict-of-Interest Policy

1. **Institutional independence**: Every seat pairs a Chair and Alternate (or Joint Co-Chairs in `T`) from distinct institutions, and at most one voting Chair across the 7 seats may share an institution with the Editor. Current primary and secondary institutional/industrial appointments are verified and dated at the time of invitation.
2. **Mandatory conflict-of-interest disclosure and item-level recusal**:
   - Before reviewing a packet, each reviewer discloses (a) co-authorship or joint benchmark/consortium leadership with the Editor (for example, shared `MLPerf` / `MLCommons` working-group leadership), (b) authorship or leadership of open-source frameworks or standards related to candidate mechanisms (for example, `PagedAttention` in `P4`, compiler stacks in `S2`, or benchmark protocols in `Q1`), and (c) commercial vendor ties.
   - A reviewer recuses from voting on the admission, preferred label, definition, or aliasing of mechanisms originating from their own publications, consortia, or commercial products (note that multi-mechanism frameworks and benchmark suites such as `vLLM` and `MLPerf` are already categorically excluded under Gate 3).
   - **Escalation when both paired reviewers are conflicted**: If both the Chair and Alternate on a seat have a disclosed conflict on an item, adjudication transfers to `V1` plus an unconflicted adjacent Group Chair.
3. **Quorum, voting, and limits on Editor overrides**:
   - **Quorum**: Release sign-off requires participation of at least `6` of the `7` voting seats, which must include both `V1` and `V2` plus all mandatory sub-area specialist reports in §2.
   - **Decision rule**: Item-level disputes not resolved within a group packet are decided by majority vote (`>= 4` of `7` seats); `1.0` release sign-off requires `>= 5` of `7` seats voting `approve`, including affirmative votes from both `V1` and `V2` within their respective scopes.
   - **Recorded Editor override rule**: While the Editor retains final repository maintenance authority during `0.x-draft`, for a `1.0` citable release the Editor **may not claim board approval** on any item where a board majority (or `V1`/`V2` on standards conformance) voted `reject`; any Editor dissent or deferred item must be published verbatim in `taxonomy/CHANGELOG.md` and `taxonomy/METHOD.md`.

---

## 4. Scoped Work Packages and Dual-Human Inter-Indexer Protocol

Board members and human indexers receive role-specific, machine-generated packets from `taxonomy/downloads/`:

1. **Domain Group Packets (`F`, `S`, `P`, `Q`, and `T1`+`T2`, plus mandatory sub-area specialists)**:
   - *Subject & Concept Inventory* (`12`–`25` subjects, `90`–`240` concepts per group): flag missing `must`-have mechanisms backed by $\ge 2$ independent `provenance_family_id` lineages (`>= 1` non-project-internal), any product/suite leak (`Gate 3`), or any home-rule misplacement.
   - *Split Hierarchy Queue (`16` edges total across the taxonomy)*: adjudicate each `specializes` edge where prior model audits split using the 3-step `all-and-some` test (`Z39.19 §8.3.1`).
   - *Zero-Warrant Queue (`27` concepts total across the taxonomy)*: confirm external literary warrant in `S1`–`S5` or recommend retirement (`retired` with `replaced_by`) or upward-posted aliasing.
2. **Vocabulary & SKOS Specialist Packets (`V1` and `V2`)**:
   - *`V1` (Z39.19 / ISO 25964 Packet)*: audit `taxonomy/METHOD.md` clause mappings against primary standard text; review the **`86` legacy compound labels** matching `\b(and|versus|vs\.)\b` (`87` when including `\bor\b`, computed from `ml-systems-taxonomy.yml`) against `Z39.19 §7`; audit legacy `aliases` across all `1,256` entry terms for inverted broader terms (`assurance case` on `mlsys.safety-case`) or method-artifact conflations (`unsafe control action` / `UCA` on `mlsys.stpa`); and sign off on the pre-registered human inter-indexer study protocol before data collection.
   - *`V2` (SKOS / RDF / Crosswalk Packet)*: audit `mlsys-taxonomy.ttl`, `mlsys-taxonomy-relations.ttl`, `mlsys-taxonomy.dcat.ttl`, URI dereferencing, conditional-edge RDF caveats, and SKOS crosswalks (`ACM CCS`, `CS2023`, `SFIA 9`, `ESCO`).
3. **Dual-Human Inter-Indexer Consistency Study (`Z39.19 §11.2.2` evaluation protocol)**:
   - **Staffing (`12` distinct domain indexers)**: Two independent human domain indexers (PhD candidates, postdocs, or staff ML systems engineers who did not author the taxonomy) are recruited for each of **six domain strata** — `F`, `S`, `P`, `Q`, `T1` (`Agentic systems`), and `T2` (`Embodied & physical systems`) — totaling **`6 × 2 = 12` distinct domain indexers** with zero personnel overlap across `T1` and `T2` (and separate from the voting board Chairs).
   - **Two-stage split discipline**:
     1. *Stage 1 (Calibration, `5` items per stratum = `30` calibration items)*: Indexers tag `5` practice items using `taxonomy/README.md` (`tagging_rules`), reconcile misunderstandings of the rules with `V1`, and freeze the written indexing guide. Calibration items are discarded from final metrics.
     2. *Stage 2 (Pre-registered Evaluation, fresh uninspected sample)*: Drawn from uninspected `reserve` / post-cutoff sources across all six strata (`F`, `S`, `P`, `Q`, `T1`, `T2`), completely disjoint from the inspected `P4`+`Q4` study pool.
   - **Scoring and pre-registered reporting**: Each indexer independently assigns `1` primary subject and `0`–`3` concepts per item (permitting subject-only tags when no concept is specific enough). The report publishes **both conditional (`both non-empty`) and all-item** `Hooper`, `Rolling`, Krippendorff's $\alpha_{\text{MASI}}$, subject Cohen's $\kappa$, 95% bootstrap confidence intervals, empty-assignment rates, leaf-concept specificity ratio, and max single-concept share ($p_{\max} \le 15\%$).
