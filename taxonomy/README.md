# ML Systems Taxonomy

A shared vocabulary for organizing, teaching, and indexing machine learning systems knowledge.

The taxonomy accompanies the [*Machine Learning Systems* textbook series](https://mlsysbook.ai/) and provides a shared vocabulary for its [chapters](../books/), [courses and teaching materials](https://mlsysbook.ai/instructors/), [labs](../labs/), and [StaffML assessment questions](../staffml/). It is also designed for reuse beyond the books. See the [repository overview](../README.md) for the complete curriculum.

**Status:** editorial draft (`0.4-draft`), open to review and contributions.

The taxonomy organizes the engineering knowledge needed to construct, evaluate, deploy, and operate machine learning systems. Its scope includes conventional prediction, generative services, agents, and learned components in physical systems. Its organizing question is: **what bodies of knowledge does an engineer need to reason about the behavior of an ML system?** Using ML to optimize other systems is an adjacent research direction, not the organizing scope here.

## Start here

- **Explore:** [Browse the taxonomy](https://mlsysbook.ai/taxonomy/), search for a concept, or follow its relationships.
- **Reuse:** [Download and cite](https://mlsysbook.ai/taxonomy/#bok-release) the vocabulary in JSON, CSV, SKOS, or GraphML. Store stable identifiers such as `mlsys.kv-cache` when tagging your material.
- **Contribute:** [Suggest a concept or correction](https://github.com/harvard-edge/cs249r_book/issues/new?template=taxonomy_feedback.yml), including the affected entry and supporting sources.
- **Maintain:** Follow the [maintenance instructions](#maintaining-the-taxonomy). See [Architecture](ARCHITECTURE.md) for file ownership and code organization.

## How this taxonomy is built

We developed the taxonomy through an iterative synthesis of educational and technical sources. A field survey, an inventory of textbook topics, and a critique of the initial vocabulary informed its organizing framework. Coverage checks, relationship audits, and classification exercises then informed successive revisions.

The development recipe is:

1. **Survey the field.** Examine course syllabi, research literature, established frameworks, standards, and textbook topics. Record the sources and define the intended users, tasks, and scope.
2. **Establish the organizing framework.** Group the knowledge into foundations, stack, lifecycle, qualities, and system types. Define the areas and subjects within each group, including their boundaries.
3. **Develop the concepts.** Give each concept a stable identifier, a preferred label, a precise definition, supporting references, and one primary subject. Distinguish synonyms from narrower terms, related ideas, and implementations.
4. **Define and review relationships.** Connect concepts using explicit relation types and state when each relationship holds. Read each connection as a complete assertion; revise or remove misleading claims.
5. **Check coverage and use.** Compare the vocabulary against research topics, teaching material, and classification tasks. Record missing concepts, ambiguous placements, and disagreements, then use those findings to refine the taxonomy. Material used to revise the taxonomy becomes construction evidence rather than an independent test.
6. **Version and maintain.** Preserve identifiers, record accepted changes and replacements, validate the source and exports, and revisit the vocabulary as evidence and feedback accumulate.

The method is informed by [ANSI/NISO Z39.19](https://www.niso.org/publications/ansiniso-z3919-2005-r2010) and [ISO 25964](https://www.niso.org/schemas/iso25964), with [W3C SKOS](https://www.w3.org/TR/skos-reference/) used for interchange. These standards guide vocabulary management and representation; they do not prescribe the subject inventory. Full standards conformance has not been established.

See [Method](METHOD.md) for source selection, decision rules, development history, evaluation methods, and evidence limitations. Independent expert validation and user studies remain outstanding; this draft does not claim complete coverage or community consensus.

## Authoritative records

| File or directory | Purpose |
|---|---|
| [taxonomy.yml](taxonomy.yml) | Canonical groups, subjects, concepts, relationships, guides, and retired identities |
| [provenance.yml](provenance.yml) | Recorded changes and supporting evidence, tied to the canonical content |
| [schema/](schema/) | Formal record definitions, controlled values, and validation rules |
| [mlsysbook-alignment.yml](mlsysbook-alignment.yml) | Connections between concepts and textbook passages |
| [coverage/](coverage/), [agreement/](agreement/) | Recorded coverage and classification evaluations; interpretation and limitations are documented in [Method](METHOD.md) |
| [reviews/](reviews/) | Editorial decisions, sources, and validation records |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

Historical study artifacts are preserved separately in [method-study/](method-study/). See [Architecture](ARCHITECTURE.md) for implementation files and generated outputs.

`taxonomy/taxonomy.yml` is the single authored source for the taxonomy. It holds the subject groups, areas, and subjects (`areas`), the concepts with their `home` subject (`concepts`), the controlled relation verbs (`relations`) and conditional relationships (`relationships`), the practitioner guides (`guides`: roles, symptoms, and learning paths), and the explicit SKOS hierarchy pilot (`skos_hierarchy`). Subject concept lists and the `all` view are derived by the compiler from `Concept.home` and `concepts`.

## Organizing frame

An ML system is a **stack**, run through a **lifecycle**, judged by **qualities**, built on shared **foundations**. Two **system types**, agentic and embodied, add engineering that has no general home. The browsing outline organizes areas, subjects, and concepts under those groups:

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

Chapters, quizzes, StaffML questions, slides, labs, and courses are tagged with the same vocabulary, so a competency is a concept or subject paired with a depth. The rules (`tagging_rules` in `taxonomy.yml`, also shipped in the JSON download and shown on the page):

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

### Classification evaluation

Records in [agreement/](agreement/) support reproducible checks of recorded tagging decisions and referenced identities. Run `./binder/binder concepts agreement --check` to recompute them. Their collection methods and interpretation are described in [Method](METHOD.md#historical-study-and-evidence-recovery). Recomputing a record does not constitute a new evaluation; material used to refine the tagging rules is construction evidence.

## Website

`taxonomy/index.qmd` follows the presentation conventions of established taxonomies (MITRE ATT&CK and ATLAS, SFIA, ESCO, CS2023) and of concept maps (Novak's labeled propositions), in the neutral mlsysbook.ai style. Color carries no category; crimson marks interaction only.

- **Concept map (home).** Opens first with the field-wide relationship figure (a circular subject map paired with an area relationship grid) and the most connected concepts. Each concept page also carries a local map: concepts as text chips, what acts on the concept on the left and what it acts on on the right, under relation headings, at full text size; selecting a chip opens that concept. The relationships table gives every link with the condition under which it holds, and "How does this relate to…" returns chains of relationships to another concept as sentences.
- **Matrix.** The whole taxonomy on one screen (`#view=matrix`): one band per group, each area a column, each subject a cell with its code. A *Show concepts* toggle lists concepts in each cell, a *Highlight* filter dims subjects that do not apply to a kind of system (generative, agentic, physical, edge, distributed, recommendation, multimodal), and an *Overlay* picker annotates cells by role, learning path, or volume. Below the concept map and matrix: *Download & cite*.
- **Entry pages.** Area, subject, and concept pages share one layout: a sidebar tree of the taxonomy with the current entry marked, a breadcrumb, the code and title, a plain definition, a facts box, and tables (subjects of an area; concepts of a subject; a concept's relationships and book passages).
- **Guides.** Ways in for people who arrive with a job rather than a concept (`guides` in `taxonomy/taxonomy.yml`): role profiles with core and working subjects, production symptoms with their likely causes, measured `quantities`, what to check, and what to change, and learning paths through the subjects. The compiler rejects any unknown subject or concept and any symptom check or lever that shares no relationship with one of the symptom's listed causes or declared `quantities`, so the guides stay tied to the graph. The same guides ship in the JSON download.
- **Glossary.** A table of every concept with its definition, search terms, and subject code, filterable by text and area.
- **Print.** *Printable outline* prints the full outline (groups, areas, subjects with definitions and concepts).

State is kept in the URL hash (`#view=matrix`, `#subject=…`, `#concept=…`, `#area=…`, `#view=glossary`, `#view=guides&role=…|symptom=…|path=…`); earlier hash formats still resolve.

## Relationships

Relationships use a controlled vocabulary of 17 relations (`relations` in `taxonomy.yml`), each with a forward and an inverse phrase so incoming links read naturally ("is reduced by"). The compiler requires every relationship to name one, requires its `kind` to agree with the relation, and rejects two relationships between the same pair of concepts. The specific `predicate` wording and the `conditions` sentence remain the detail. Each relationship's `references` name the reading basis of the source concept's family, not a citation that proves the proposition; the field `reference_role` records that. Current inventory counts are generated by `binder concepts check` and shown in the explorer and exports. Sparse or disconnected semantic relationships are reported for review rather than rejected; every concept still requires a browsing home. Edge counts never justify adding unsupported assertions. Historical audit summaries remain in the changelog. Current accepted corrections record exact changes and supporting sources in the [editorial reviews](reviews/README.md); retained relationships remain draft propositions with stated conditions.

## Codes, downloads, and citation

Every group, area, and subject carries a readable code: a group letter (F, S, P, Q, T; Lifecycle uses P so a code never reads as a StaffML depth level), an area number, and a subject number, so `P4.3` is Lifecycle › Deployment & serving › its third subject. The compiler requires codes to be unique and nested. Codes are for people and may be renumbered between versions; concept identities such as `mlsys.kv-cache` are permanent and are what tools should store.

`binder concepts build` writes the public release to `taxonomy/downloads/` from the same compiled records as the page:

| File | Format | Contents |
|---|---|---|
| `mlsys-taxonomy-subjects.csv` | CSV | One row per subject (`code, group, area, subject, definition, concepts, concept_ids, taught_in`) |
| `mlsys-taxonomy-concepts.csv` | CSV | One row per concept (`id, term, aliases, definition, subject_code, subject, area, group, background, taught_in, added_in, literary_warrant, corpus_chapters, organizational_warrant, user_warrant_proxy`) |
| `mlsys-taxonomy.json` | JSON | The full outline, concepts (with candidate book passages), relation vocabulary, typed relationships, tagging rules, placement examples, guides (roles, symptoms, learning paths), declared gaps, retired identities, machine-readable `content_revision` (`urn:sha256:...`), and `Source YAML SHA-256` provenance note when amended in place |
| `mlsys-taxonomy.linkml.yml` | LinkML (YAML) | LinkML schema defining classes, slots, controlled lists, and W3C `skos:`, `dct:`, and `owl:` URI mappings |
| `mlsys-taxonomy.schema.json` | JSON Schema | Validation schema compiled from `mlsys-taxonomy.linkml.yml` |
| `mlsys-taxonomy.ttl` | SKOS (RDF Turtle) | Groups, areas, and subjects as `skos:Collection` with `skos:notation` codes and scope notes; concepts as `skos:Concept` with history notes; `specializes` as `skos:broader`; retired identities marked deprecated; `dct:identifier "urn:sha256:..."` and `Source YAML SHA-256` provenance note in `dct:description` when amended in place |
| `mlsys-taxonomy-relations.ttl` | OWL (RDF Turtle) | The extension vocabulary: controlled relation properties under `skos:broader` or `skos:related`, plus the condition metadata property |
| `mlsys-taxonomy.dcat.ttl` | DCAT (RDF Turtle) | Dataset and distribution metadata (including `dct:identifier "urn:sha256:..."` for the canonical YAML source) |
| `CITATION.cff` | Citation File Format | Citation metadata for the current version |
| `mlsys-concept-graph.graphml` | GraphML | The concept graph for Gephi, yEd, Cytoscape, or a graph database |
| `mlsys-taxonomy.bib` | BibTeX | Citation for the current version |

The page's *Print / PDF* action prints the expanded outline. The taxonomy release (identifiers, labels, definitions, relationships, and guides) is licensed CC BY 4.0, so courses, standards bodies, and companies can reuse and adapt it with attribution; the book prose keeps its own license.

## Maintaining the taxonomy

Follow the [construction and maintenance protocol](METHOD.md#construction-and-maintenance-protocol). For each proposed change:

1. Search the existing vocabulary across areas, including labels, aliases, definitions, and retired identities.
2. Record the source evidence, intended meaning, and indexing use case. Distinguish a separate concept from a synonym, narrower entry term, or related mechanism.
3. Review the definition, placement, aliases, and every affected relationship. Record disagreements and the editorial disposition.
4. Apply accepted changes through a verified provenance-preserving workflow. Preserve permanent identities and retirement history.
5. Run the compiler, build, coverage and agreement checks, and affected downstream checks. Inspect the rendered website and exports.
6. Record the version and change rationale. Approve the concrete release before deployment.

Draft corrections now use a small public review command. Start from the [worked review](reviews/mechanism-boundaries/accepted-sparse-attention.json), recording the original record, proposed fields, source support, rationale, reviewer, and affected relationships:

```bash
python3 binder/tools/audit/review_taxonomy.py REVIEW.json /tmp/taxonomy-preview
# Inspect the candidate and originals, then apply with a fresh output directory:
python3 binder/tools/audit/review_taxonomy.py REVIEW.json /tmp/taxonomy-applied --apply
```

The command validates the candidate and preserves the provenance chain. It supports corrections, additions, typed entry-term decisions, moves, explicit splits and retirements, and draft version transitions. It rejects stale records and identity reuse. See the [operation guide](reviews/lifecycle.md) for the JSON contract, impact requirements, and versioned archive command.

Accepted changes require supporting sources, recorded editorial decisions, and automated integrity checks. Review requirements depend on the impact of the change; see [Method](METHOD.md#maintenance-cycle) for the maintenance policy.

## Related work

[W3C SKOS](https://www.w3.org/TR/skos-primer/) supplies representation conventions for controlled vocabularies, hierarchies, associations, and collections. It does not supply the subject inventory. [CS2023 knowledge areas](https://csed.acm.org/knowledge-areas/) help identify adjacent computing foundations and boundaries. [Software Engineering for Machine Learning](https://www.microsoft.com/en-us/research/publication/software-engineering-for-machine-learning-a-case-study/) supplies an empirical workflow perspective, while [The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) informs production-readiness and testing questions. The [NIST AI RMF](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf) contributes a risk perspective; its functions are not a linear technical curriculum.

Related taxonomies already exist. [A Taxonomy of ML for Systems Problems](https://research.google/pubs/a-taxonomy-of-ml-for-systems-problems/) organizes ML applied to systems problems. [An architectural perspective on MLOps](https://doi.org/10.1016/j.infsof.2026.108029) organizes operational architecture. The proposed framework needs a documented comparison with these neighboring scopes before making a novelty claim. The full design-reference list is generated from the source alongside the written framework.

## Build and validation

Run these commands from the repository root after changing the authored sources:

```bash
./binder/binder concepts check
./binder/binder concepts build
./binder/binder concepts build --check
```

The checks validate record structure, identities, references, provenance, and generated outputs. The website, downloads, and StaffML lookup data share the same canonical source. See [Architecture](ARCHITECTURE.md#safe-maintenance) for regression tests, website checks, and implementation details, including [conditional relationships in RDF](ARCHITECTURE.md#reading-conditional-relationships-in-rdf).

The `aliases` compatibility field contains search terms, which are not necessarily synonyms. The explorer identifies them as search terms; SKOS alternative labels require an explicit synonym or acronym decision. See [Method](METHOD.md#standards-assessment) for representation rules and review limits.
