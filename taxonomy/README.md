# Machine learning systems: a candidate subject framework

Status: editorial draft, version `0.4-draft`. This document specifies a proposal for review, rather than a claim of field completeness or community consensus.

The framework organizes the engineering knowledge needed to construct, evaluate, deploy, and operate machine learning systems. Its scope includes conventional prediction, generative services, agents, and learned components in physical systems. Its organizing question is: **what bodies of knowledge does an engineer need to reason about the behavior of an ML system?** Using ML to optimize other systems is an adjacent research direction, not the organizing scope here.

## How this taxonomy is built

The taxonomy follows an explicit, repeatable engineering workflow grounded in published controlled-vocabulary standards (`taxonomy/METHOD.md`, operationalized in `.claude/workflows/taxonomy-refresh.md` and `taxonomy/method-study/synthesized-method.md`):

1. **Standards adopted.** Construction, term choice, compound factoring, and equivalence/hierarchy control follow [ANSI/NISO Z39.19-2005 (R2010)](https://groups.niso.org/higherlogic/ws/public/download/6484/Guidelines%20for%20the%20Construction,%20Format,%20and%20Management%20of%20Monolingual%20Controlled%20Vocabularies.pdf) and the structural model of ISO 25964-1:2011; machine-readable representation follows [W3C SKOS](https://www.w3.org/TR/skos-reference/) with a declared OWL extension for 17 typed relations; publication metadata follows DCTERMS and W3C DCAT 3; and indexing consistency is measured with Hooper's measure, Rolling's measure, Krippendorff's $\alpha_{\text{MASI}}$, and Cohen's $\kappa$ paired with anti-gaming specificity bounds ($p_{\max} \le 15\%$).
2. **Enumerated five-stratum sampling frame.** Completeness and candidate discovery are anchored on a written sampling frame (`taxonomy/method-study/sampling-frame.md`) spanning five strata: university graduate courses (`S1`), systems and ML conference programs (`S2`), engineering and safety/governance standards (`S3`), practitioner role profiles and untagged StaffML interview questions (`S4`), and MLSysBook chapter contracts (`S5`). Source units are partitioned by `provenance_family_id` into `60%` `dev` (visible during construction), `25%` `held_out_test` (sealed until post-freeze evaluation), and `15%` `reserve`.
3. **Incremental refresh over `0.4-draft`, verified by three model families.** Rather than rebuilding from scratch, each refresh runs three independent model families (`claude-opus-5-5`, `gpt-6-astra` / `gpt-6.1-sol`, `gemini-4-argon-high`), deterministic cross-check reconciliation (`reconcile.py`), LinkML + `binder concepts check` compilation, an independent necessity/publishability audit, and a blind held-out evaluation. `METHOD.md` itemizes every applicable Z39.19 and SKOS clause as met, partial, or open, including the main remaining open step before a citable `1.0` release: review by the external editorial board (`taxonomy/method-study/candidate-experts.md`) and a human inter-indexer trial.

## Authoritative records

| Record | Question it answers | Authoritative source |
|---|---|---|
| LinkML schema | What classes, slots, controlled lists, and URI mappings (`skos:`, `dct:`, `owl:`) govern the taxonomy? | `taxonomy/schema/mlsys-taxonomy.linkml.yml` (`UnifiedTaxonomyModel`) |
| Taxonomy source (single source of truth) | What belongs in this engineering field, what does each concept mean, where is its home subject, how does it affect other ideas under stated conditions, and which subjects and concepts matter for a given role, production symptom, or learning path? | `taxonomy/ml-systems-taxonomy.yml` |
| Book alignment | Where does MLSysBook discuss an idea? | `taxonomy/mlsysbook-alignment.yml` |
| Competency specification, proposed | What can a learner demonstrate under a stated task, context, and evidence requirement? | Separate task records, not yet adopted |

`taxonomy/ml-systems-taxonomy.yml` is the single authored source for the taxonomy. It holds the 5 subject groups, 20 areas, and 101 subjects (`areas`), the 878 concepts with their `home` subject (`concepts`), the 17 controlled relation verbs (`relations`) and 2,016 conditional relationships (`relationships`), the practitioner guides (`guides`: roles, symptoms, and learning paths), and the explicit SKOS hierarchy pilot (`skos_hierarchy`). Subject concept lists and the `all` view are derived by the compiler from `Concept.home` and `concepts`.

## Organizing frame

An ML system is a **stack**, run through a **lifecycle**, judged by **qualities**, built on shared **foundations**. Two **system types**, agentic and embodied, add engineering that has no general home. Version `0.4-draft` arranges 20 subject areas and 101 subjects (rows) under those five groups, with 878 concepts:

| Group | Areas |
|---|---|
| Foundations | Workloads & requirements · Models & architectures · Quantitative system models |
| Stack | Hardware & accelerators · Frameworks, compilers & runtimes · Parallelism & distributed execution · Datacenter infrastructure & fleet |
| Lifecycle | Data engineering · Training & adaptation · Model evaluation & testing · Deployment & serving · MLOps & operations |
| Qualities | Performance, efficiency & benchmarking · Reliability & fault tolerance · Security & privacy · Safety & assurance · Responsible AI · Sustainability & cost |
| System types | Agentic systems · Embodied & physical systems |

The groups correspond to the layer, activity, concern, and context facets. A single tree necessarily flattens that faceted space, so each concept's one home exists for maintenance and browsing, while facets and secondary links carry the other legitimate placements. The placement rules are recorded in the framework `rationale`:

- **Home rule.** A mechanism that implements a layer of the stack or a stage of the lifecycle lives in that Stack or Lifecycle subject, even when it exists for speed or cost. A mechanism whose only purpose is one quality, and which applies across layers, lives in that Quality, beside the measurement of that quality. General quantitative system models live in Foundations, while measurement of a specific quality lives in that Quality, except metrics defined by one lifecycle stage (serving latency targets), which live with that stage. Worked examples (`placement_examples`, also shown on the page):

  | Concept | Home | Why |
  |---|---|---|
  | KV cache | Generative inference | Part of how decoding works, though it exists for speed |
  | KV-cache quantization | Model compression | Only purpose is efficiency; the technique spans weights, activations, and caches |
  | Speculative decoding | Generative inference | Changes how the serving stage generates tokens |
  | Activation recomputation | Training memory | Implements a training-stage memory plan |
  | Carbon-aware scheduling | Energy & carbon | Only purpose is lower carbon, across training, inference, and data jobs |
  | Kernel fusion | ML compilers | Implements the compiler layer |
  | DP-SGD | Privacy & DP | Only purpose is privacy |
- **System-type admission test.** A system type earns an area only when, after placement, it keeps core concepts with no general home. Recommendation, multimodal, generative, edge, and distributed settings are contexts, shown as lenses.
- **Background tier.** Concepts developed outside ML systems that ML systems reasoning depends on (loss functions, consensus, feedback control) carry `scope: background`. The tag records origin, not importance: a background concept is not optional and implies no lower depth.
- **Uneven subjects.** Subjects differ in size because fields do; a three-concept subject is not a weaker one, and size is not a weight.
- **Identity references.** Includes and excludes cite owning rows as `[[row-id]]`; the compiler rejects unknown identities.
- **Declared gaps.** Topics a reviewer should expect but this version does not yet cover are recorded as `gaps` with a `topic` and a subject row, and published on the page.

Using ML to optimize systems is an adjacent field and out of scope. Chapter pointers (`book_chapters`) name where MLSysBook teaches a subject at chapter grain; they are reading pointers, not alignment or depth claims.

## Tagging with the taxonomy

Chapters, quizzes, StaffML questions, slides, labs, and courses are tagged with the same vocabulary, so a competency is a concept or subject paired with a depth. The rules (`tagging_rules` in `ml-systems-taxonomy.yml`, also shipped in the JSON download and shown on the page):

- Tag the concepts an item centrally teaches or tests, preferring one and adding a second or third only when the item cannot be answered without it; never prerequisites or passing mentions.
- A subject alone is a valid, complete tag when the item is broad or no concept is specific enough.
- When no concept fits, tag the closest subject and record the missing idea as a proposed concept for the next version.
- Store identities (`mlsys.kv-cache`, `llm-inference`), never codes.
- Do not tag a workload-class concept (Large language model, Transformer) only because an LLM is involved.
- A retired identity resolves to its `replaced_by`; when a concept is split, existing tags resolve to the replacement until they are re-reviewed.
- A concept that seems to fit two subjects takes its home under the home rule; related subjects are navigation, not alternative tags.
- Tag the mechanism, not the platform: cloud, edge, mobile, and TinyML are contexts. Edge & embedded hardware and On-device deployment are subjects only when the item is about the device class or about deploying to it.
- Tag a calculation by what it is about; Performance models, Workload characterization, and Profiling are subjects only when the item teaches the method.
- When an item asks for a technique's effect, the technique is the subject and the affected quantity may be a second concept.
- Depth is recorded separately (for example, StaffML levels L1 to L6+).

### Tagging agreement

A vocabulary that things are labeled with must label the same item the same way for different taggers. `taxonomy/agreement/staffml-300-2026-10.yml` records two independent taggers, `gpt-6.1-sol` and `claude-opus-5-5`, tagging the same 300 StaffML questions (sampled across all topics and tracks) without seeing each other's work, and `./binder/binder concepts agreement --check` recomputes the numbers and fails if a recorded tag cites a retired identity.

| Run | Subject | Area | Main concept | Same concept set | Compatible |
|---|---|---|---|---|---|
| 1 (first rules) | 78.7% (κ 0.78) | 82.3% (κ 0.81) | 74.6% | 37.1% | 93.0% |
| 2 (rules revised) | 83.7% (κ 0.83) | 87.0% (κ 0.86) | 80.4% | 55.9% | 94.3% |
| Held out (fresh 300) | 80.7% (κ 0.80) | 84.7% (κ 0.83) | 74.5% | 45.0% | 92.7% |

*Compatible* means the taggers chose the same subject, shared a concept, or one chose a concept homed in the other's subject. Run 1 disagreements had three causes: filing by platform instead of mechanism (mobile and edge questions agreed only 65% and 69% of the time), filing calculations under the modeling method, and filing a technique's effect under the effect. The three rules above, a preference for one concept, and 13 concepts the taggers proposed came from them; mobile and edge rose to 78% and 76%. Run 2 is not an independent test, because the rules were revised on the same sample; the held-out row is, measured on 300 different questions after the rules were fixed (`staffml-300-heldout-2026-10.yml`). Model taggers are not people, so human taggers are the next measurement.

The same rules then tagged all 10,716 StaffML questions (`staffml/vault/data/taxonomy-tagging-2026-10.yaml` records the method and the review queue of disagreements). Questions left without a concept proposed what was missing; the recurring proposals became eight concepts (Transformer variants, data transfer cost, duty cycling, priority inversion, randomized smoothing, sensor interconnect bandwidth, finite buffers, asynchronous on-device inference), and those questions were retagged.

## Website

`taxonomy/index.qmd` follows the presentation conventions of established taxonomies (MITRE ATT&CK and ATLAS, SFIA, ESCO, CS2023) and of concept maps (Novak's labeled propositions), in the neutral mlsysbook.ai style. Color carries no category; crimson marks interaction only.

- **Concept map (home).** Opens first with the field-wide relationship figure (a circular map across the 5 groups, 20 areas, and 101 subjects, paired with a 20 × 20 area matrix) and the most connected concepts. Each concept page also carries a local map: concepts as text chips, what acts on the concept on the left and what it acts on on the right, under relation headings, at full text size; selecting a chip opens that concept. The relationships table gives every link with the condition under which it holds, and "How does this relate to…" returns chains of relationships to another concept as sentences.
- **Matrix.** The whole taxonomy on one screen (`#view=matrix`): one band per group, each area a column, each subject a cell with its code. A *Show concepts* toggle lists concepts in each cell, a *Highlight* filter dims subjects that do not apply to a kind of system (generative, agentic, physical, edge, distributed, recommendation, multimodal), and an *Overlay* picker annotates cells by role, learning path, or volume. Below the concept map and matrix: *Download & cite*.
- **Entry pages.** Area, subject, and concept pages share one layout: a sidebar tree of the taxonomy with the current entry marked, a breadcrumb, the code and title, a plain definition, a facts box, and tables (subjects of an area; concepts of a subject; a concept's relationships and book passages).
- **Guides.** Ways in for people who arrive with a job rather than a concept (`guides` in `taxonomy/ml-systems-taxonomy.yml`): fourteen role profiles with core and working subjects, twenty-three production symptoms with their likely causes, what to check, and what to change, and seven learning paths through the subjects. The compiler rejects any unknown subject or concept and any symptom check or lever that shares no relationship with one of the symptom's causes, so the guides stay tied to the graph. The same guides ship in the JSON download.
- **Glossary.** A table of every concept with its definition, aliases, and subject code, filterable by text and area.
- **Print.** *Printable outline* prints the full outline (groups, areas, subjects with definitions and concepts).

State is kept in the URL hash (`#view=matrix`, `#subject=…`, `#concept=…`, `#area=…`, `#view=glossary`, `#view=guides&role=…|symptom=…|path=…`); earlier hash formats still resolve.

## Relationships

Relationships use a controlled vocabulary of 17 relations (`relations` in `ml-systems-taxonomy.yml`), each with a forward and an inverse phrase so incoming links read naturally ("is reduced by"). The compiler requires every relationship to name one, requires its `kind` to agree with the relation, and rejects two relationships between the same pair of concepts. The specific `predicate` wording and the `conditions` sentence remain the detail. Each relationship's `references` name the reading basis of the source concept's family, not a citation that proves the proposition; the field `reference_role` records that. In version `0.4-draft` the graph has 2,016 relationships; every concept has at least two, the graph is connected, and the mean shortest path between concepts is about five links. A literal-reading audit of every relationship in 0.2 corrected or removed 76 that read false or backwards; every relationship added in 0.3 and 0.4 was read the same way by independent reviewers in other model families before it was accepted, and the 0.4 refresh added three independent audits and a blind held-out test after the merge (see the changelog).

## Codes, downloads, and citation

Every group, area, and subject carries a readable code: a group letter (F, S, P, Q, T; Lifecycle uses P so a code never reads as a StaffML depth level), an area number, and a subject number, so `P4.3` is Lifecycle › Deployment & serving › its third subject. The compiler requires codes to be unique and nested. Codes are for people and may be renumbered between versions; concept identities such as `mlsys.kv-cache` are permanent and are what tools should store.

`binder concepts build` writes the public release to `taxonomy/downloads/` from the same compiled records as the page:

| File | Format | Contents |
|---|---|---|
| `mlsys-taxonomy-subjects.csv` | CSV | One row per subject (`code, group, area, subject, definition, concepts, concept_ids, taught_in`) |
| `mlsys-taxonomy-concepts.csv` | CSV | One row per concept (`id, term, aliases, definition, subject_code, subject, area, group, background, taught_in, added_in, literary_warrant, corpus_chapters, organizational_warrant, user_warrant_proxy`) |
| `mlsys-taxonomy.json` | JSON | The full outline, concepts (with candidate book passages), relation vocabulary, typed relationships, tagging rules, placement examples, guides (roles, symptoms, learning paths), declared gaps, and retired identities |
| `mlsys-taxonomy.linkml.yml` | LinkML (YAML) | LinkML schema defining classes, slots, controlled lists, and W3C `skos:`, `dct:`, and `owl:` URI mappings |
| `mlsys-taxonomy.schema.json` | JSON Schema | Validation schema compiled from `mlsys-taxonomy.linkml.yml` |
| `mlsys-taxonomy.ttl` | SKOS (RDF Turtle) | Groups, areas, and subjects as `skos:Collection` with `skos:notation` codes and scope notes; concepts as `skos:Concept` with history notes; `specializes` as `skos:broader`; retired identities marked deprecated |
| `mlsys-taxonomy-relations.ttl` | OWL (RDF Turtle) | The extension vocabulary: 17 relation properties under `skos:broader` or `skos:related` |
| `mlsys-taxonomy.dcat.ttl` | DCAT (RDF Turtle) | Dataset and distribution metadata |
| `CITATION.cff` | Citation File Format | Citation metadata for the current version |
| `mlsys-concept-graph.graphml` | GraphML | The concept graph for Gephi, yEd, Cytoscape, or a graph database |
| `mlsys-taxonomy.bib` | BibTeX | Citation for the current version |

The page's *Print / PDF* action prints the expanded outline. The taxonomy release (identifiers, labels, definitions, relationships, and guides) is licensed CC BY 4.0, so courses, standards bodies, and companies can reuse and adapt it with attribution; the book prose keeps its own license.

## Maintaining the taxonomy

The taxonomy is meant to be complete enough to rely on and cheap to keep current. Single additions and periodic refreshes follow the same standard-grounded gates (`taxonomy/METHOD.md`, `.claude/workflows/taxonomy-refresh.md`):

1. **Search all 20 areas and verify warrant before adding.** Before proposing any concept for a subject, search `taxonomy/downloads/mlsys-taxonomy-concepts.csv` across all 20 areas by label, aliases, and definition words. If an existing concept in another area already covers the mechanism (for example, `mlsys.memory-planning` in `S2` or `mlsys.selective-prediction` in `P3`), add an `alias` or a `secondary_concepts` link on the subject row rather than minting a duplicate concept. Every new concept requires literary warrant from at least two independent `provenance_family_id` lineages (at least one in `S1`–`S4`, and at least one source containing explanatory prose defining the mechanism; a syllabus or checklist topic heading can corroborate a prose source as the second lineage, but bare headings alone with zero prose sources cannot mint a concept), must pass the Z39.19 §6.3.3 product/SKU exclusion gate (products, engines, benchmarks, and SKUs become `aliases`), must pass the Z39.19 §7 compound-term and `metric × workload` factoring check (rejecting `inference throughput` or `inference energy cost` splits unless they define a stage-specific structural measure such as `TTFT`, `TPOT`, or `goodput`), and must pass the 4-part Concept-vs-Alias Differential Indexing Witness test (`Z39.19 §8.2`; tightly coupled phases such as `prefill phase` / `decode phase` and internal sub-steps/artifacts such as `unsafe control action` on `mlsys.stpa` become `aliases` on the parent concept, whereas distinct methodologies such as `FMEA`, `Fault tree analysis`, and `STPA` under `Hazard analysis` remain separate `specializes` concepts).
2. **Add or edit.** A new concept is a single record in `taxonomy/ml-systems-taxonomy.yml` (`id`, `label`, `home`, `family`, `definition`, optional `aliases`) plus at least two relationships in `relationships` using the controlled relations; subject concept lists and the `all` view derive automatically from `home` (and the order of concepts within each subject row follows their order in `concepts:`). Every `specializes` edge must pass the 3-step `all-and-some` test (`Z39.19 §8.3.1`), and every relationship must read true as a plain sentence ("Source verb Target") under its stated `conditions`.
3. **Build.** `./binder/binder concepts build` checks the source taxonomy, book alignment, coverage checklists, coverage records, tagging-agreement records, and compiled releases against `taxonomy/schema/mlsys-taxonomy.linkml.yml`. It also verifies that hierarchical relations (`specializes`, `part-of`, `requires`) form no cycles, that `specializes` and associative relations are disjoint (`SKOS S27`), that verb and predicate direction agree, that symptom buckets do not overlap, that every subject appears in at least one role, and that homes, codes, references, relation verbs, concept pairs, and retired identities are consistent. It then writes the page data, public downloads, and StaffML taxonomy labels (`staffml/app/src/data/mlsys-taxonomy.json`).
4. **Measure.** `./binder/binder concepts coverage --check` recomputes coverage against every recorded checklist and fails if a judgment cites a concept or subject that no longer exists. Add a new checklist (for example, a new year of conference papers) as another file in `taxonomy/coverage/`, marked `held_out: true` when it did not inform the design. `./binder/binder concepts agreement --check` does the same for recorded tagging-agreement runs; add a fresh sample as a new record rather than retagging an old one.
5. **Record.** Bump `version` in `taxonomy/ml-systems-taxonomy.yml` and `taxonomy/schema/mlsys-taxonomy.linkml.yml`, add an entry to `taxonomy/CHANGELOG.md`, and retire rather than delete any withdrawn identity.
6. **Publish.** Pushes to `dev` run `🧭 Taxonomy · ✅ Validate (Dev)` and `🧭 Taxonomy · 👁️ Preview (Dev)` automatically (`taxonomy-validate-dev.yml`, `taxonomy-preview-dev.yml`). Once validate is green, dispatch `🧭 Taxonomy · 🚀 Publish (Live)` (`.github/workflows/taxonomy-publish-live.yml` with `confirm=PUBLISH`) to deploy `https://mlsysbook.ai/taxonomy/` on its own without touching the landing page or other subsites, or run `🌐 Landing Site · 🚀 Publish (Live)` (`.github/workflows/site-publish-live.yml`) to deploy the entire unified site.

A label or alias may name only one concept; the compiler rejects a name shared by two, so a glossary search or a tagger always lands on one entry.

**Periodic refresh (`.claude/workflows/taxonomy-refresh.md`).** A few times a year, and before any citable release, the taxonomy is refreshed against new `dev` evidence from the five-stratum sampling frame while keeping `held_out_test` sealed. Three model families (`claude-opus-5-5`, `gpt-6-astra` / `gpt-6.1-sol`, `gemini-4-argon-high`) independently propose and cross-check candidate updates; `reconcile.py` verifies that no `fix` or `reject` verdict is silently dropped; three independent auditors check necessity, provenance-family independence, and anti-splitting; and two blind evaluators score coverage and inter-indexer agreement (`Hooper`, `Rolling`, $\alpha_{\text{MASI}}$, $\kappa$, with max single-concept share $p_{\max} \le 15\%$) on the held-out slice. Model reviewers are not people; external domain-expert board review (`taxonomy/method-study/candidate-experts.md`) and human indexer trials remain the stronger test required before a `1.0` release.

## Authoring and illustration tools

The authoring system and student interface serve different users. [LinkML](https://linkml.io/) (`taxonomy/schema/mlsys-taxonomy.linkml.yml`) defines the formal YAML metamodel for the taxonomy, mapping classes and slots to W3C `skos:`, `dct:`, and `owl:` URIs and compiling directly to JSON Schema (`mlsys-taxonomy.schema.json`). [VocBench](https://vocbench.uniroma2.it/) supports collaborative management of SKOS vocabularies and OWL ontologies. [Protégé](https://protege.stanford.edu/) supports OWL ontology development and reasoning.

For this repository, LinkML-validated YAML (`binder/cli/core/taxonomy_schema.py`) plus ontological graph checks in `binder concepts check` keeps the framework beside the book contracts and makes revisions inspectable in Git. The SKOS hierarchy comes only from `specializes` edges, audited with the Z39.19 all-and-some test; subjects stay collections and never become broader terms. A future editor integration should import/export through that LinkML and SKOS contract, rather than create a second independently edited truth.

The student illustrations use native HTML/SVG and generated, focused Graphviz relationship drawings. This supports readable typography, in-place selection, keyboard controls, deterministic static exports, and the same source data without a dependency on an external visualization service. Cytoscape remains available for graph interactions where it improves the learning task.

## Relation to established work

[W3C SKOS](https://www.w3.org/TR/skos-primer/) supplies representation conventions for controlled vocabularies, hierarchies, associations, and collections. It does not supply the subject inventory. [CS2023 knowledge areas](https://csed.acm.org/knowledge-areas/) help identify adjacent computing foundations and boundaries. [Software Engineering for Machine Learning](https://www.microsoft.com/en-us/research/publication/software-engineering-for-machine-learning-a-case-study/) supplies an empirical workflow perspective, while [The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) informs production-readiness and testing questions. The [NIST AI RMF](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf) contributes a risk perspective; its functions are not a linear technical curriculum.

Related taxonomies already exist. [A Taxonomy of ML for Systems Problems](https://research.google/pubs/a-taxonomy-of-ml-for-systems-problems/) organizes ML applied to systems problems. [An architectural perspective on MLOps](https://doi.org/10.1016/j.infsof.2026.108029) organizes operational architecture. The proposed framework needs a documented comparison with these neighboring scopes before making a novelty claim. The full design-reference list is generated from the source alongside the written framework.

## Competency design probes

A subject label identifies knowledge; a competency requires observable engineering work. Four draft probes illustrate the difference:

| Task | Connected knowledge | Evidence to assess |
|---|---|---|
| Decide whether quantization improves a constrained inference workload | Profiling, workload, bandwidth, arithmetic intensity, quantization, metrics, SLOs | Reproducible baseline, measured bottleneck, quality comparison, decision with stated limits |
| Recover distributed training under an explicit failure model | Parallelism, training state, checkpoints, consistent cuts, artifact identity, collectives | Checkpoint manifest, injected-failure/restart logs, restored-state checks, guarantee boundaries |
| Configure a retrieval-assisted service under quality and capacity constraints | Retrieval, ANN, embeddings, benchmark design, hallucination, admission, tail latency, goodput, cost | Retrieval and answer evaluation, workload traces, failure accounting, transfer-case reassessment |
| Evaluate timing and fallback behavior of a learned physical component | Operating domain, observation age, feedback delay, execution time, WCET, runtime assurance | Contained experiment, timing evidence, fallback behavior, limits of the tested operating domain |

These are proposed assessment designs, not claims that existing chapters assess those tasks. A later AI engineering profile could select relevant knowledge and tasks with a declared role, autonomy, constraints, and evidence rubric. Reading a chapter, visiting a map, or recognizing concepts does not demonstrate competence. The [CS2023 competency introduction](https://csed.acm.org/wp-content/uploads/2024/04/1.4-Introduction-to-Competency-Framework.pdf) provides a useful task-and-context distinction; [SFIA responsibility levels](https://sfia-online.org/en/sfia-9/responsibilities) describe professional responsibility and must not be converted into a student mastery score.

## Repeatable build and review

Run `./binder/binder concepts build` after editing the authored sources. Run `./binder/binder concepts check` and `./binder/binder concepts build --check` to validate identities, assignments, references, section-backed passage links, and generated-artifact freshness. The build exports `framework-data.json`, `framework-table.csv`, `framework-reference.md`, `field-data.json`, `taxonomy.ttl`, `taxonomy-reference.md`, all eight release files in `taxonomy/downloads/`, and the StaffML taxonomy bundle (`staffml/app/src/data/mlsys-taxonomy.json`). The figures, website, and StaffML app use that same compiled source. Existing chapter concept contracts continue to supply the book graph.

Adding a field concept requires a candidate home decision; the compiler rejects unresolved or duplicate homes. Secondary placements do not inflate the inventory. Passage links resolve against actual chapter sections and remain related-passage links with unassessed teaching depth. The pipeline deliberately avoids inferring relationships from embedding proximity. Embeddings can later propose candidates for review, while accepted definitions, placements, and relationships remain explicit versioned records.

Before a citable `1.0` release, the project needs a frozen scope, author/editor decisions, a literature crosswalk, assertion-level technical review, independent human classification exercises, student navigation studies, version and change policy, and an archival publication decision. Of the four method gaps identified at the start of `0.4-draft`, three are now resolved in `taxonomy/METHOD.md` and `taxonomy/method-study/`: (1) an enumerated five-stratum sampling frame (`S1`–`S5`) with provenance-family cluster splits is recorded in `taxonomy/method-study/sampling-frame.md`; (2) the concept-versus-alias boundary and compound-term splitting rules (`Z39.19 §7`, `§8.2`) are operationalized via the 4-part Differential Indexing Witness test, coupled-phase anti-splitting, and compiled `metric × workload` checks; and (3) held-out coverage and indexing consistency are evaluated by two independent judges with ordinal expressibility scoring and anti-gaming specificity bounds (`Hooper`, `Rolling`, $\alpha_{\text{MASI}}$, $\kappa$, $p_{\max} \le 15\%$). The fourth item — end-to-end subject review by the external domain-expert editorial board (`taxonomy/method-study/candidate-experts.md`) and a human inter-indexer trial — remains the primary gate before a `1.0` citable release. Book mapping needs a separate coverage/depth audit. No DOI, external endorsement, license change, or completed human validation is implied by this prototype. Independent model critiques informed this iteration; they are design feedback rather than community consensus.
