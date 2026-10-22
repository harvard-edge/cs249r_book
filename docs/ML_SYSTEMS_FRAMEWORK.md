# Machine learning systems: a candidate subject framework

Status: editorial draft, version `0.2-draft`. This document specifies a proposal for review, rather than a claim of field completeness or community consensus.

The framework organizes the engineering knowledge needed to construct, evaluate, deploy, and operate machine learning systems. Its scope includes conventional prediction, generative services, agents, and learned components in physical systems. Its organizing question is: **what bodies of knowledge does an engineer need to reason about the behavior of an ML system?** Using ML to optimize other systems is an adjacent research direction, not the organizing scope here.

## Four connected records

| Record | Question it answers | Authoritative source |
|---|---|---|
| Subject framework | What belongs in this engineering field, and where can I find it? | `shared/concepts/ml-systems-framework.yml` |
| Concept vocabulary and relationships | What does an idea mean, and how does it affect other ideas under stated conditions? | `shared/concepts/ml-systems-field.yml` |
| Book alignment | Where does MLSysBook discuss an idea? | `shared/concepts/mlsysbook-alignment.yml` |
| Competency specification, proposed | What can a learner demonstrate under a stated task, context, and evidence requirement? | Separate task records, not yet adopted |

The existing `ml-systems-taxonomy.yml` contains a small, explicit SKOS hierarchy pilot. It does not silently acquire broader/narrower assertions from the new subject collections. A predictable browsing location is useful without asserting that every collection is a semantic superclass.

## Organizing frame

An ML system is a **stack**, run through a **lifecycle**, judged by **qualities**, built on shared **foundations**. Two **system types**, agentic and embodied, add engineering that has no general home. Version `0.2-draft` arranges 20 subject areas and 98 subjects (rows) under those five groups:

| Group | Areas |
|---|---|
| Foundations | Workloads & requirements · Models & architectures · Quantitative system models |
| Stack | Hardware & accelerators · Frameworks, compilers & runtimes · Parallelism & distributed execution · Datacenter infrastructure & fleet |
| Lifecycle | Data engineering · Training & adaptation · Model evaluation & testing · Deployment & serving · MLOps & operations |
| Qualities | Performance, efficiency & benchmarking · Reliability & fault tolerance · Security & privacy · Safety & assurance · Responsible AI · Sustainability & cost |
| System types | Agentic systems · Embodied & physical systems |

The groups correspond to the layer, activity, concern, and context facets. A single tree necessarily flattens that faceted space, so each concept's one home exists for maintenance and browsing, while facets and secondary links carry the other legitimate placements. The placement rules are recorded in the framework `rationale`:

- **Mechanism first.** A concept lives in the Stack or Lifecycle row that engineers its mechanism whenever one exists (KV caching in serving, kernel fusion in compilers), even when it exists for performance.
- **Qualities hold measurement and cross-layer techniques.** A Quality area holds measurement, trade-off reasoning, and techniques that act across several layers at once, such as compression.
- **System-type admission test.** A system type earns an area only when, after placement, it keeps core concepts with no general home. Recommendation, multimodal, generative, edge, and distributed settings are contexts, shown as lenses.
- **Background tier.** Concepts imported from general ML, robotics, distributed systems, site reliability, and safety engineering carry `scope: background`. They remain because ML systems concepts depend on them.
- **Identity references.** Includes and excludes cite owning rows as `[[row-id]]`; the compiler rejects unknown identities.
- **Declared gaps.** Topics a reviewer should expect but this version does not yet cover are recorded as `gaps` with a `topic` and a subject row, and published on the page.

Using ML to optimize systems is an adjacent field and out of scope. Chapter pointers (`book_chapters`) name where MLSysBook teaches a subject at chapter grain; they are reading pointers, not alignment or depth claims.

## Website

`site/explore/index.qmd` follows the presentation conventions of established taxonomies (MITRE ATT&CK and ATLAS, SFIA, ESCO, CS2023) and of concept maps (Novak's labeled propositions), in the neutral mlsysbook.ai style. Color carries no category; crimson marks interaction only.

- **Matrix (home).** The whole taxonomy on one screen: one band per group, each area a column, each subject a cell with its code. A *Show concepts* toggle lists concepts in each cell, and a *Highlight* filter dims subjects that do not apply to a kind of system (generative, agentic, physical, edge, distributed, recommendation, multimodal). Below the matrix: *Download & cite*.
- **Entry pages.** Area, subject, and concept pages share one layout: a sidebar tree of the taxonomy with the current entry marked, a breadcrumb, the code and title, a plain definition, a facts box, and tables (subjects of an area; concepts of a subject; a concept's relationships and book passages).
- **Concept map.** Each concept page carries a local map: concepts as text chips, what acts on the concept on the left and what it acts on on the right, under relation headings, at full text size; selecting a chip opens that concept. The relationships table gives every link with the condition under which it holds, and "How does this relate to…" returns chains of relationships to another concept as sentences.
- **Glossary.** A table of every concept with its definition, aliases, and subject code, filterable by text and area.
- **Print.** *Printable outline* prints the full outline (groups, areas, subjects with definitions and concepts).

State is kept in the URL hash (`#subject=…`, `#concept=…`, `#area=…`, `#view=glossary`); earlier hash formats still resolve.

## Relationships

Relationships use a controlled vocabulary of 17 relations (`relations` in `ml-systems-field.yml`), each with a forward and an inverse phrase so incoming links read naturally ("is reduced by"). The compiler requires every relationship to name one, requires its `kind` to agree with the relation, and rejects two relationships between the same pair of concepts. The specific `predicate` wording and the `conditions` sentence remain the detail. In version `0.2-draft` the graph has 1,392 relationships; every concept has at least two, the graph is connected, and the mean shortest path between concepts is about five links. A literal-reading audit of every relationship corrected or removed 76 that read false or backwards.

## Codes, downloads, and citation

Every group, area, and subject carries a readable code: a group letter (F, S, L, Q, T), an area number, and a subject number, so `L4.3` is Lifecycle › Deployment & serving › its third subject. The compiler requires codes to be unique and nested. Codes are for people and may be renumbered between versions; concept identities such as `mlsys.kv-cache` are permanent and are what tools should store.

`binder concepts build` writes the public release to `site/explore/downloads/` from the same compiled records as the page:

| File | Format | Contents |
|---|---|---|
| `mlsys-taxonomy-subjects.csv` | CSV | One row per subject: code, group, area, definition, concepts, chapters |
| `mlsys-taxonomy-concepts.csv` | CSV | One row per concept: identity, term, aliases, definition, home code, background flag |
| `mlsys-taxonomy.json` | JSON | The full outline, concepts, relation vocabulary, and typed relationships |
| `mlsys-taxonomy.ttl` | SKOS (RDF Turtle) | Groups, areas, and subjects as `skos:Collection` with `skos:notation` codes; concepts as `skos:Concept`; relationships as properties under `skos:related` |
| `mlsys-concept-graph.graphml` | GraphML | The concept graph for Gephi, yEd, Cytoscape, or a graph database |
| `mlsys-taxonomy.bib` | BibTeX | Citation for the current version |

The page's *Print / PDF* action prints the expanded outline. The release currently carries the book's CC BY-NC-SA 4.0 license; a more permissive license would ease adoption by other courses and organizations and is an open decision.

## Maintaining the taxonomy

The taxonomy is meant to be complete enough to rely on and cheap to keep current. Each update follows the same loop:

1. **Add or edit.** A new concept is one record in `ml-systems-field.yml` (identity, label, definition, family, optional aliases) plus at least two relationships using the controlled relations, and its identity added to one subject's `concepts` in `ml-systems-framework.yml`. Read every new relationship as a plain sentence ("Source verb Target") before keeping it.
2. **Build.** `./binder/binder concepts build` validates homes, codes, references, relation vocabulary, duplicate pairs, and retired identities, then regenerates the page data and downloads.
3. **Measure.** `./binder/binder concepts coverage --check` recomputes coverage against every recorded checklist and fails if a judgment cites a concept or subject that no longer exists. Add a new checklist (for example, a new year of conference papers) as another file in `shared/concepts/coverage/`, marked `held_out: true` when it did not inform the design.
4. **Record.** Bump `version` in `ml-systems-framework.yml`, add an entry to `docs/ML_SYSTEMS_TAXONOMY_CHANGELOG.md`, and retire rather than delete any withdrawn identity.

## Authoring and illustration tools

The authoring system and student interface serve different users. [VocBench](https://vocbench.uniroma2.it/) supports collaborative management of SKOS vocabularies and OWL ontologies. [Protégé](https://protege.stanford.edu/) supports OWL ontology development and reasoning. Both are credible options if external editors need a dedicated semantic authoring environment. They do not determine the ML systems categories or make an ontology accurate automatically.

For this repository, reviewed YAML plus schema validation keeps the framework beside the book contracts and makes revisions inspectable. The existing SKOS export remains restricted to explicitly authored hierarchy assertions. A future editor integration should import/export through that contract, rather than create a second independently edited truth. Adopting OWL would require an actual need for formal classes, properties, constraints, and reasoning; drawing a connected diagram alone does not establish that need.

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

Run `./binder/binder concepts build` after editing the authored sources. Run `./binder/binder concepts check` and `./binder/binder concepts build --check` to validate identities, assignments, references, section-backed passage links, and generated-artifact freshness. The build exports `framework-data.json`, `framework-table.csv`, and `framework-reference.md`. The figures and website use that same framework. Existing chapter concept contracts continue to supply the book graph.

Adding a field concept requires a candidate home decision; the compiler rejects unresolved or duplicate homes. Secondary placements do not inflate the inventory. Passage links resolve against actual chapter sections and remain related-passage links with unassessed teaching depth. The pipeline deliberately avoids inferring relationships from embedding proximity. Embeddings can later propose candidates for review, while accepted definitions, placements, and relationships remain explicit versioned records.

Before a citable release, the project needs a frozen scope, author/editor decisions, a literature crosswalk, assertion-level technical review, independent classification exercises, student navigation studies, version and change policy, and an archival publication decision. Book mapping needs a separate coverage/depth audit. No DOI, external endorsement, license change, or completed human validation is implied by this prototype. Independent model critiques informed this iteration; they are design feedback rather than community consensus.
