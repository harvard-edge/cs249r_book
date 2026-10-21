# Machine learning systems: a candidate subject framework

Status: editorial draft, version `0.1-draft`. This document specifies a proposal for review, rather than a claim of field completeness or community consensus.

The framework organizes the engineering knowledge needed to construct, evaluate, deploy, and operate machine learning systems. Its scope includes conventional prediction, generative services, agents, and learned components in physical systems. Its organizing question is: **what bodies of knowledge does an engineer need to reason about the behavior of an ML system?** Using ML to optimize other systems is an adjacent research direction, not the organizing scope here.

## Four connected records

| Record | Question it answers | Authoritative source |
|---|---|---|
| Subject framework | What belongs in this engineering field, and where can I find it? | `shared/concepts/ml-systems-framework.yml` |
| Concept vocabulary and relationships | What does an idea mean, and how does it affect other ideas under stated conditions? | `shared/concepts/ml-systems-field.yml` |
| Book alignment | Where does MLSysBook discuss an idea? | `shared/concepts/mlsysbook-alignment.yml` |
| Competency specification, proposed | What can a learner demonstrate under a stated task, context, and evidence requirement? | Separate task records, not yet adopted |

The existing `ml-systems-taxonomy.yml` contains a small, explicit SKOS hierarchy pilot. It does not silently acquire broader/narrower assertions from the new subject collections. A predictable browsing location is useful without asserting that every collection is a semantic superclass.

## Organizing rule

A subject area groups engineering knowledge with related mechanisms, design decisions, and failure modes. A subarea names a body of knowledge and defines its boundary. Activities such as deployment, system layers such as runtime, and concerns such as security are independent facets. They are not competing branches of one tree.

Each current concept has one candidate home to make editorial maintenance predictable. Explicit secondary associations preserve legitimate connections across subjects. For example, retrieval belongs in inference and services while also connecting to data management. Neither placement dictates a course sequence or a prerequisite. Facet assignments are editorial hypotheses about relevance. Unassigned context facets mean unassessed relevance, not inapplicability.

The source contains definitions, includes/excludes boundaries, reading warrants, stable IDs, and candidate inventory gaps. A reading warrant identifies the scope of material used in designing a row. It does not establish exact semantic equivalence, endorsement, or assertion-level evidence. Empty concept inventories identify work to investigate; they do not establish that the books omit a subject.

## Visual design study

The Quarto page `site/explore/index.qmd` reads one generated data payload. Its taxonomy and concept-map modes share the same selection and site navigation. `taxonomy-study.qmd` preserves the former preview URL using the same interface body. The taxonomy has two display layouts:

- **Branching tree:** the field expands into subject areas, subareas, and concepts. Selection reveals detail in place; the overview remains recoverable without opening another page.
- **Nested map:** stable rectangular regions reveal progressively finer detail, with a persistent route back to the whole field. Geometry does not encode importance.

The **Concept map** mode includes every current concept and authored relationship. Selecting a concept preserves the map and opens its definition. An optional neighborhood view shows direct connections. The full drawing is a reference overview; fitting it on screen does not make every label readable.

The systems-perspective and activity-matrix views remain static reference alternatives. Subjects may be relevant to several system perspectives; these placements do not assert exclusive membership, a universal architecture, or causal flow.

A selected concept shows its definition and labeled connections in the same workspace. Following a connection changes the selection in place. Related book passages are optional behind one disclosure. The reference matrix, CSV, and written definitions remain available as scholarly and editorial artifacts without competing for the learner's attention in the main canvas.

A focused concept map answers a different question. It shows labeled, conditional relationships around a selected concept, with navigation to related concepts and book passages. It should accompany the atlas rather than replace the classification with an unreadable network of every edge.

The implementation uses HTML and SVG for inspectable subject views. [D3 partition layouts](https://d3js.org/d3-hierarchy/partition) are an option for a later semantic-zoom hierarchy; [Cytoscape.js](https://js.cytoscape.org/) remains appropriate for relationship graphs. A more elaborate renderer cannot repair unclear categories. These prototypes prioritize readable labels, stable locations, explicit semantics, and equivalents that do not require pointing at a graphic.

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
