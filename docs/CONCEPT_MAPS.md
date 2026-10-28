# Chapter concept maps and the explorer

The chapter `*_concepts.yml` files are the source of truth for teaching contracts
and book-derived maps. The independent field reference model and book alignment
are described below. The chapter files' existing
`primary_concepts`, `secondary_concepts`, `claims`, `touches`, and `previews`
retain their editorial ownership meaning. An optional `learner_graph` extension
unpacks those teaching claims into learner concepts and explanatory relationships.

The same records generate chapter viewers, website maps, per-volume catalogs,
and the series teaching overview. JSON, DOT, SVG, and standalone HTML are build
artifacts, not alternative authoring sources.

## Student navigation and information architecture

The primary object is **one concept map per volume**. Chapters are
locations and highlights inside that graph, not separate ownership trees.
A student selects a volume, sees its broader conceptual structure, selects a
chapter to highlight its concepts, and follows labeled connections into other
chapters without replacing or rearranging the map. Zoom-to-chapter and
zoom-to-connections retain the same graph geometry. Concept selection and
chapter highlights persist in the expanded tab's URL.

The default volume maps unpack selected teaching claims into smaller learner
concepts and source-backed relationships across chapters. They are a first editorial draft of the major connections, not an
exhaustive graph of all the existing teaching goals. The catalog retains every
owned teaching goal; no goal is silently promoted into a causal relationship
from ownership, chapter co-occurrence, or a similarity score.

The four scopes are the model, fleet, trajectory, and physical action. They are
independent books rather than a required reading sequence. Each concept retains
its volume and owning chapter. Cross-volume equivalence requires separate
editorial authoring and is not inferred from matching names.

Two granularities share the same pipeline and viewer. Authored learner volume
views are primary; the earlier argument inventories remain generated for reference
and are excluded from the student map selector when a learner volume view exists:

- `explorer_graph` connects existing owned teaching arguments across chapters.
  Each edge cites exact existing chapter claims and is initially a draft.
- `learner_graph` defines finer concepts and question or volume views, with exact
  chapter-section excerpts. The original focused serving maps remain available.

`taxonomy/atlas.yml` stores only volume navigation metadata. It does not own
concepts or edges. Source excerpts establish provenance and freshness, not
semantic entailment. Review must confirm the predicate, direction, and
qualification of each proposed edge before its status becomes `reviewed`.

### Authoring cross-chapter propositions

Add this optional extension under the owning chapter's `concept_map`. A source
must be an existing concept owned by that chapter. The target must be owned by
the named chapter in the same volume. No concept text is duplicated as a second
definition. Evidence quotes are checked against the referenced `claims` values.

```yaml
explorer_graph:
  schema_version: 1
  relationships:
    - source: Kernel fusion
      predicate: reduces traffic through
      target:
        chapter: 11_hw_acceleration
        concept: Operand-supply hierarchy
      explanation: Fusion avoids materializing intermediate tensors across the memory hierarchy.
      conditions: Applies to compatible operators whose intermediates need not be preserved separately.
      status: draft
      evidence:
        - chapter: 07_frameworks
          concept: Kernel fusion
          quote: Most elementwise and normalization operators are memory-bound
        - chapter: 11_hw_acceleration
          concept: Operand-supply hierarchy
          quote: Delivered throughput is set by where operands live
```

The volume graph joins these chapter-owned propositions by canonical identity.
Changing an endpoint name or its evidence invalidates the build instead of
quietly publishing a stale relationship. New concepts appear in the catalog
immediately; appearing in the volume map requires an authored proposition.

## Commands

From the repository root:

```bash
./binder/binder concepts check --vol1
./binder/binder concepts refresh --vol1
./binder/binder concepts build --vol1
./binder/binder concepts build --check
```

- `check` validates authored records without generating assets.
- `refresh` inventories definition callouts, indexed definitions, and glossary terms and reports changed or invalid
  relationship evidence in `books/_build/concepts/review.json`. It does not edit
  chapter YAML. Pending evidence changes persist across refreshes until the
  authored relationship is updated. This is a deterministic candidate inventory;
  it does not yet use embeddings or a model to infer new relationships.
- `build` validates, lays out the maps, and writes generated assets. It resolves
  the series graph before writing any assets, so invalid authoring leaves the
  previous output intact. `--volN` scopes embedded chapter viewers; the shared
  series catalog and website map assets are always refreshed together.
- `build --check` reports missing or stale generated artifacts without writing them.

Install PyYAML and Graphviz (`dot`) before building. The site workflows and book
container definitions declare these dependencies. Exact SVG layout depends on the
installed Graphviz version and fonts; repeated builds in one environment are stable.

Every volume's HTML pre-render hook calls the same generator. An HTML-only Lua
filter inserts a collapsed explorer immediately after the chapter's learning
objectives when the chapter participates in a volume map. The embedded viewer
opens the same volume graph with the chapter highlighted. The chapter box includes an **Expand interactive map** button that opens a full
browser tab and preserves the selected map and concept. The viewer provides
zoom-in, zoom-out, and fit controls. The prose, PDF, and EPUB remain unchanged. Chapters that own no concepts render normally.

## Authoring the first map in a chapter

Keep the existing contract intact. Add this extension under `concept_map`:

```yaml
learner_graph:
  schema_version: 1
  concepts:
    - id: vol1.serving.queueing-delay
      label: Queueing delay
      teaches: Queueing sets headroom
      summary: Requests wait when arrivals compete for busy resources.
      section: sec-model-serving-utilizationlatency-relationship-a2f0
    - id: vol1.serving.utilization
      label: Utilization
      teaches: Queueing sets headroom
      summary: Utilization compares offered work with service capacity.
      section: sec-model-serving-utilizationlatency-relationship-a2f0
  relationships:
    - id: vol1.serving.wait-utilization
      source: vol1.serving.queueing-delay
      predicate: rises nonlinearly with
      target: vol1.serving.utilization
      explanation: Waiting grows rapidly as the system approaches saturation.
      conditions: Stable M/M/1 model; production workloads require measurement.
      status: draft
      evidence:
        section: sec-model-serving-utilizationlatency-relationship-a2f0
        quote: latency grows nonlinearly
  maps:
    - id: vol1.serving.queueing
      focus_question: Why does saturation increase request waiting?
      description: Connect arrival pressure, service capacity, and delay.
      concepts:
        - vol1.serving.queueing-delay
        - vol1.serving.utilization
      relationships:
        - vol1.serving.wait-utilization
```

Each learner concept has one authoritative definition. `teaches` must name an
existing owned primary or secondary concept in that same chapter; multiple
learner concepts may unpack one teaching claim. Reuse concept IDs rather than
copying definitions into another chapter. IDs are stable identifiers, not labels
that should change with wording.

A relationship has one authoritative home: the chapter containing its supporting
section. Its endpoints may refer to concepts owned by other chapters. Its quote
must occur in the supporting section, with whitespace normalized. Section-level
links connect the map directly to the teaching. The generator validates that the
evidence exists; an editor must still judge that it supports the relationship
and that the conditions are sufficient. `draft` and `reviewed` are author-supplied
states; the pipeline never promotes a relationship automatically.

Maps reference concept and relationship IDs. They do not redefine them. Every
relationship endpoint must appear in its map. A chapter may supply several
focused maps, each answering a different question. A broader map can reuse the
same records. Mentioning a concept or reusing it does not establish a prerequisite.

## Optional understanding checks

A map can contain `checks`. Each check references a concept in that map:

```yaml
checks:
  - concept: vol1.serving.queueing-delay
    question: What can happen when arrival pressure increases?
    choices:
      - Requests spend longer waiting for service.
      - Memory bandwidth necessarily increases.
    answer: 0
    explanation: At fixed service capacity, increased arrivals raise utilization.
```

`answer` is the zero-based choice index. These are explanatory checks, not a
mastery score or a replacement for the chapter exercises. No learner data is sent
to a service or stored.

## Adding or revising material

1. Update the manuscript and its ownership contract as appropriate.
2. Run `concepts refresh` to find definition candidates and affected evidence.
3. Edit the optional learner records where the new material earns a place in a map.
4. Run `concepts check` and `concepts build`.
5. Render the chapter, inspect its map and reading links, and review the generated
   series overview. Inspect SVGs after layout changes.

The detailed response-time and LLM-capacity views remain available in Model Serving,
with shared concepts owned by Hardware Acceleration and Model Compression.
The series catalog aggregates teaching goals and declared reuse from all chapter contracts;
it does not imply that detailed semantic maps have been authored for every chapter.

## Generated locations

- `taxonomy/catalog.json`: series ownership/reuse catalog, volume metadata, and map manifest.
- `taxonomy/atlas.html`: offline series preview with embedded catalog and detailed maps.
- `taxonomy/volumes/volN.json`: volume teaching overview.
- `taxonomy/maps/<map-id>/`: map JSON, DOT, and SVG.
- `books/volN/<chapter>/concepts-generated/`: portable local chapter viewer.
- `books/_build/concepts/chapters.json`: manifest used by the HTML injection filter.

The website and chapter viewer share the same JavaScript/CSS and generated map
payloads. Chapter viewers embed their data and SVGs, so their local `index.html`
can also be opened directly from disk without starting a server. Graphviz places labels and edges; Cytoscape supplies graph-neighborhood
queries for highlighting. Map selections and selected concepts have shareable URLs.
The source excerpt checks, graph integrity tests, and build freshness check cover
mechanical consistency, not pedagogical completeness or technical entailment.

## Review findings and the field-map design

The October 20 review used independent AI roles for ML systems architecture,
student learning, and curriculum alignment, followed by cross-review of source
corrections. It was not consultation with human experts or a student study.
Reviewers rejected treating chapter coverage or graph connectivity as evidence
of conceptual completeness. All authored relationships remain drafts.

The review removed unsupported registry/scheduler, data-selection/training-law,
batching/memory-hierarchy, phase-margin/stopping-delay, and other bridges. It
corrected fusion to reducing total dispatch overhead rather than cost per dispatch,
distinguished valid collectives from correct inputs, and distinguished context
partitioning from increased KV payload. No original teaching claims or ownership
assignments changed. Remaining argument-title nodes and generic qualifications
still need a systematic editorial pass; these corrections do not certify every edge.

### Target model: concepts, teaching occurrences, and views

A mature field graph needs short concepts with definitions, aliases, and stable
IDs; explicit propositions with conditions and direct manuscript evidence; and
teaching occurrences that locate those concepts in each independent book.
The chapter ownership contract remains authoritative for what is taught. A learner
concept unpacks a claim rather than replacing its ownership entry. Maps select
from this authored graph and answer questions; they do not define a second graph.

Current `learner_graph` already represents finer concepts and section evidence.
Current `explorer_graph` represents broad teaching arguments. It is a transitional
overview, not the completed atomic field ontology. Similar words across volumes
must not create automatic identity or prerequisite links. Cross-volume comparisons
need explicit authoring and assumptions; the book's own reading path stays local.

Candidate exploration families, to be checked against definitions, equations,
worked examples, quizzes, and exercises rather than chapter quotas:

| Family | Examples of finer concepts | Changes of context across the series |
| --- | --- | --- |
| Work and resource limits | Operations, bytes, bandwidth, capacity, utilization | Single-system execution; fleet capacity; context residency; physical renewal |
| Representations and behavior | Labels, precision, inductive bias, distribution shift | Dataset specification; distributed updates; trajectory data; action labels |
| Time and scheduling | Arrival rate, service time, queueing delay, deadline, jitter | Serving; stragglers; tool waits; observation age and response bounds |
| State and reuse | Working set, residency, version, cache validity, reconstruction | Activations; checkpoints; KV blocks and durable records; dated observations |
| Failure and recovery | Detection, rollback, compensation, irreversibility | Model release; fleet recovery; uncertain tool effects; stopping and fallback |
| Authority and boundaries | Permission, mediation, lease, isolation, admission | Release gates; allocation; tool grants; actuation permission |
| Evidence and acceptance | Baseline, estimand, uncertainty, invariant, acceptance criterion | Benchmark validity; assurance; task grading; closed-loop safety evidence |

These families are navigation candidates, not mandatory nodes, a validated taxonomy,
or claims that the current diagrams cover each family adequately.

Volume maps remain the main entry points. Chapter selection highlights concepts
**taught here**, not every concept required to study that chapter. Focused question
maps expose finer explanations. The generated `detail_maps` links join overview
and learner concepts by exact owning chapter and teaching-goal identity; no
similarity matching is used. Selecting a question changes to its explanatory map;
**Back to the volume map** restores the original selected concept. Concepts with
no authored detail view have no such link. All chapter viewers carry the authored
maps so these routes also work offline.

### Repeatable expansion and acceptance

1. Inventory candidate concepts from definition callouts, glossary/index terms,
   equations, figures, worked examples, and exercises. Retain exact source spans.
2. Normalize names and aliases; separate mechanisms, quantities, policies, and
   guarantees. Admit a concept when it supports an explanation or engineering task.
3. Author propositions from the actual explanation. Endpoint quotations establish
   provenance, but not entailment. Label synthesis/inference explicitly and supply
   its warrant. Never invent a bridge merely to connect components.
4. Check coverage by engineering task and family. Record unmapped concepts and
   missing explanations separately from chapter representation.
5. Independently review correctness, visible qualifications, and learner usefulness;
   revise and cross-review. AI critique remains distinct from human approval.
6. Regenerate and verify source freshness, IDs, links, layout, and navigation.
7. Before claiming learning benefit, ask students to explain a trade-off, predict
   an intervention, and locate background, comparing against ordinary navigation.

`refresh` currently implements the definition inventory and evidence-change
tracking, not this entire extraction/review workflow. Glossary/exercise extraction,
shared cross-volume concept identity, complete family coverage, and version-bound
human review provenance remain future work. Embeddings may propose candidates;
they must never publish semantic or prerequisite edges automatically.

### Companion activities

Optional `explorer_graph.resources` maps existing owned teaching goals to lists:

```yaml
resources:
  Roofline and ridge point:
    - title: Hello, Roofline
      path: mlsysim/docs/tutorials/00_hello_roofline.qmd
      kind: simulation
      status: early-release
```

Paths must identify existing repository files and stay inside the repository.
Kinds are `implementation`, `simulation`, `lab`, `slides`, or `hardware`. Status is
`ready`, `early-release`, or `skeleton`; readiness is an editorial assertion, not
inferred from file existence. The current small selection links to repository
source on `dev`, not an assumed published/runnable lesson. Skeletons explicitly
say **Planned activity — source outline only**. Links are generated for overview
nodes and all learner nodes unpacking that goal. Resources are supporting actions,
not conceptual arrows. The sample connects quantization to TinyTorch, roofline and
KV cache to MLSysIM, and enforcement to a clearly labeled planned physical lab.
Future slide integration should reuse the existing validated lecture manifest in
`slides/tools/lectures.py`, rather than creating another chapter-to-deck mapping.

Concept-map design references: Novak and Cañas's
[theory and construction guidance](https://cmap.ihmc.us/docs/theory-of-concept-maps.php)
and [propositions](https://cmap.ihmc.us/docs/Proposition.php). They support the focus
question and meaningful labeled-proposition criteria; they do not validate this
project's particular graph or establish its educational effectiveness.

## Atomic volume views and the next coverage pass

The student entry points are now the four `volN.field` maps authored under
`learner_graph.maps` in each opening chapter. They reuse concepts from the other
chapter YAMLs, including the original detailed serving maps. Only one authored
`scope: volume` map is permitted per book, and its concepts must stay inside that
book. Chapter viewers start on this map with the current chapter located.
The legacy `.overview` argument diagrams remain build artifacts; they are no
longer the primary student views when a field map exists.

```yaml
maps:
  - id: vol1.field
    scope: volume
    volume: vol1
    focus_question: How do data, learning, and machine limits shape a deployable ML system?
    description: Connect behavior, resource limits, measurement, and operational feedback.
    concepts: [vol1.data.dataset, vol1.model.behavior]
    relationships: [vol1.field.relation-0]
```

This is map membership, not duplicate concept authoring. Concept `aliases` are
optional lists of search names. Alias matches never merge identities. Primary
volume views can reuse concepts and propositions from focused question views.
The sidebar's detail links resolve exact chapter/teaching-goal identity; returning
restores the selected volume concept. Direct focused-view URLs also have a route
to the same volume's main map.

Chapter highlighting uses the original ownership/reuse contract. Teal marks
concepts taught here. Blue dashed outlines mark mapped concepts that this chapter
explicitly touches or previews; they are not inferred prerequisites. Zoom-to-chapter
includes both sets without changing the underlying geometry. A chapter can have
no represented concepts: it remains selectable, and the zero count is visible.

Every map JSON now includes `coverage`, derived from its selected concept IDs and
exact owned teaching goals. The volume viewer exposes represented goals, chapter
locations, and primary goals outside the current view. Multiple learner concepts
may unpack one goal. These are view-selection counts, not a measure of field
completeness. Source quotation presence and teaching-goal coverage still do not
prove conceptual adequacy or educational effectiveness.

The independent AI cross-reviews narrowed incorrect universal claims, distinguished
training batches from serving batches, separated workload rates from policy
mechanisms, and replaced weak mention-based evidence with explanatory paragraphs.
All propositions remain drafts. No human expert approval or student study is
implied by these review rounds.

The broader coverage check uses external teaching scopes as comparison points:

| Comparison source | Mechanisms checked in the draft | Remaining areas to expand |
| --- | --- | --- |
| [Stanford CS329S](https://stanford-cs329s.github.io/2021/syllabus.html) | Data, evaluation, deployment, monitoring, feedback | Objective/proxy mismatch, calibration and uncertainty |
| [Stanford CS336, Spring 2025](https://cs336.stanford.edu/spring2025/index.html) | Resource accounting, kernels, parallelism, scaling, data, inference | Expert/sequence parallelism, deeper optimization and statistical-efficiency mechanisms |
| [Berkeley LLM Agents, Fall 2024](https://llmagents-learning.org/f24) | Tools, retrieval, infrastructure, evaluation, safety, collaboration | Planner/search methods, retrieval methods, evaluation statistics |
| [MIT Underactuated Robotics](https://underactuated.mit.edu/index.html) | Dynamics, constraints, estimation, learning/control boundaries | Detailed control and estimation mechanisms, embodiment transfer |

This comparison is a selective sufficiency check, not endorsement by those courses,
a formal syllabus-to-concept audit, or evidence that all of their topics belong in
these books. The books' own scope and teaching contracts govern admission.

The next useful expansions are explanatory neighborhoods that support specific
tasks, rather than more arrows to increase density:

- Model scope: objective mismatch, gradient noise, convergence, uncertainty, and calibration.
- Fleet scope: critical-path stragglers, checkpoint interval, routing, autoscaling, and facility overhead.
- Trajectory scope: retrieval/planning methods, KV allocation, evaluator statistics, and detailed adaptation mechanisms.
- Physical-action scope: estimator dynamics, control constraints, contact behavior, and embodiment transfer.

`refresh` now discovers review candidates from indexed definitions and the
volume glossary as well as definition callouts, suppressing existing labels and
aliases. It retains source locations or glossary definitions and never edits
YAML or publishes an inferred relationship. Candidate discovery, curated graph
admission, independent critique, rendering, and learner testing remain distinct
steps in the repeatable workflow.

## Field reference model and book alignment

The field landscape is independent of the book-derived volume maps. Its source
is `taxonomy/ml-systems-field.yml`: stable `mlsys.*` identities,
definitions, navigation families, per-concept applicability, external reading
references, conditioned propositions, and selected views. A global concept has
no chapter owner. The chapter YAML contracts remain authoritative for what the
books teach; they are not definitions of the field's boundary.

`taxonomy/mlsysbook-alignment.yml` supplies a separate many-to-many overlay
of explicitly authored field-concept-to-book-concept links. These are related
passages, not inferred synonyms. The compiler resolves the exact existing learner
ID, chapter section, URL, and current section hash. Deleted identities fail the
build. An unmapped concept remains in the field model and means **no link
recorded yet**, never **absent from the book**. Alignment depth is currently
`not-assessed`; this first overlay does not certify adequacy.

Field concepts have independent scope labels. Generative-model techniques can
belong to the serving family without becoming universal ML systems core.
Retrieval, idempotency, and compensation retain their cross-domain identities;
agent applications do not take ownership of them. Basic mathematical and
programming background is a stated boundary rather than an attempted inventory
of adjacent disciplines.

Field relationships carry direction, predicate, kind, conditions, reading basis,
and draft status. Curriculum references justify topic selection; they do not
prove each proposition. All first-model relationships are explicitly editorial
synthesis. The validator rejects unsupported `reviewed` labels until a
version-bound review contract is implemented. Prerequisites and task-specific
learning routes require a separate authored overlay; no relation is inferred
from name similarity, adjacency, chapter order, or connectivity.

The same `binder concepts check` and `build` pipeline validates the field model.
`build --check` detects stale field data and diagrams as well as book maps.
Generated `field-data.json`, `field-all.svg`, `field-overview.svg`, their DOT
sources, and `field-preview.html` are ignored build artifacts. Invalid field
records fail before existing outputs are replaced. No external model or network
request is needed during a build.

The website page is `taxonomy/field.qmd`, linked from the book explorer.
The standalone local preview is `taxonomy/field-preview.html`. Both use the
same data and SVG geometry. The viewer supports background drag, wheel panning,
modified-wheel zoom, keyboard pan/zoom, family focus, full-model search, book-link
highlighting, and concept-to-concept exploration. HTML explanations and the
concept directory provide a text alternative to SVG interaction. Downloaded SVGs
are static field views; this does not enroll them into the PDF/EPUB books.

Focused concept views lay out an explicitly authored concept and its immediate
neighbors separately for readable labels. The neighborhood contains only
existing concepts and existing propositions. **Back to landscape** restores the
previous overview or full-map viewport. This deliberate change of view differs
from a book overlay, which highlights links without rearranging the landscape.

This remains a selective first field reference draft. Further work includes
independent source-passage verification for individual propositions, additional
concepts and relationships justified by engineering tasks, curated learning
routes, deeper book alignments, and human/student evaluation. Node counts,
connectedness, reference counts, and AI review agreement do not establish
completeness or learning benefit.

The student-facing explorer defaults to links across all books and routes reading
into MLSysBook sections. External reading references remain in the authoring
registry for editorial review and are not rendered as student destinations.
The independent concept identities still support cross-chapter navigation.

## Taxonomy design proposal

Status: design draft, not an implemented taxonomy or a reviewed field standard.

The proposed resource is **MLSysBook Taxonomy of Machine Learning Systems**.
It organizes the subject independently of chapter order while routing students
into the books. Its scope is engineering systems that learn from data: defining
objectives, acquiring and representing data, learning, executing models,
distributing work, serving requests, evaluating behavior, and operating systems
under resource and responsibility constraints. Agent and physical systems are
explicit application extensions. Adjacent mathematics, statistics, programming,
and control topics enter where needed to explain a systems decision; this is
not a replacement taxonomy for those entire disciplines.

### Standards and subject references

Use [W3C SKOS](https://www.w3.org/TR/skos-reference/) for the vocabulary model:
concept identity, preferred labels, aliases, definitions, broader/narrower
relationships, collections, and mappings between vocabularies. SKOS supplies
representation semantics, not an ML systems topic inventory. The existing YAML
registry remains the proposed authoring source; SKOS/RDF would be a generated
exchange format. This export and its additional validation are not implemented.

Use [CS2023 knowledge areas](https://csed.acm.org/knowledge-areas/) as a broad
curriculum comparison across AI, architecture, data management, systems,
parallel/distributed computing, security, and software engineering. Use the
existing registry's CMU Deep Learning Systems, Stanford CS329S, Full Stack Deep
Learning, and specialist references to investigate domain coverage. These are
comparison sources, not endorsements, proof of completeness, or substitutes
for reviewing individual concept definitions. External references serve the
editorial process; student destinations remain MLSysBook resources.

### Separate hierarchy, facets, and propositions

The current `families` are navigation groupings. They must not be automatically
exported as broader concepts: grouping membership alone establishes no broader
meaning. Review candidate topics and their narrower concepts explicitly. Allow
multiple broader concepts where justified, and record a placement rationale.
Check that the hierarchy is acyclic and that its meaning is consistent.

Keep facets separate from broader/narrower links. Candidate facets include
lifecycle activity, execution environment, model regime, and application context.
For example, distributed execution is relevant to both training and serving;
physical-system context cuts across data, evaluation, and deployment. These
facets should reuse concept IDs or controlled facet values rather than duplicate
concept definitions. They are proposed dimensions, not approved assignments.

Keep existing technical propositions separate: quantization changes numerical
precision, changes memory traffic, and requires reassessing evaluation metrics.
None of these arrows is a taxonomy parent. Prerequisites likewise require their
own authoring and assumptions rather than inference from hierarchy or proximity.

### Authoring and change contract

Extend the existing concept record rather than creating a second inventory.
A proposed concept record includes its stable ID, preferred label, genuine
aliases, definition, scope note, broader IDs, collection/facet memberships,
editorial references, and review status. The field registry owns these subject
records. Chapter YAML continues to own teaching commitments. The alignment
registry owns links to book sections; future slide/lab links require explicit
resource records. Resource locations never determine concept identity.

Adding a concept requires a distinct meaning, a useful systems question, a
scope boundary, and an editorial placement review. Renaming preserves the ID;
merging or retiring a concept requires a recorded replacement and migration of
incoming links. Review claims should be bound to a version and reviewer record.
AI suggestions and similarity searches may propose changes but do not approve
identities, hierarchy, prerequisites, or coverage.

The proposed build checks unique IDs and preferred labels, alias collisions,
missing targets, self-links, hierarchy cycles, duplicate assertions, and broken
resource identities. Generate inverse narrower links from authored broader
links rather than maintaining both. Preserve the existing conditional technical
relationships. If RDF exchange is added, validate the export against SKOS rules
and a project application profile; SHACL is an established option for RDF
constraints: https://www.w3.org/TR/shacl/.

### Publication and maintenance

Generate an expandable topic browser, the existing focused relationship maps,
and a written taxonomy reference from the same approved records. Publish scope,
definitions, inclusion rules, version history, and known boundaries alongside
the taxonomy. Do not label it the first or definitive ML systems taxonomy
without a separate literature review. Keep draft proposals distinguishable from
reviewed releases.

Start with repository YAML and the existing build pipeline. Evaluate VocBench
if collaborative visual editing becomes necessary; establish a single direction
of synchronization before introducing another editor. A graph database or OWL
reasoner is not required for the initial browsing and navigation tasks.

The next implementation slice is one reviewed branch demonstrating hierarchy,
facets, conditioned relationships, and book links together. Evaluate it through
student tasks such as locating an explanation, identifying a neighboring topic,
and navigating between chapters before classifying the full registry. This
proposal changes no live student navigation or concept ownership.

### Implemented taxonomy pilot

The authored pilot is `taxonomy/ml-systems-taxonomy.yml`. It reuses field
concept IDs and defines two additional subject topics, with explicit draft
broader/narrower placements for parallel execution strategies and learning
paradigms. Learning paradigms may overlap; their sibling placement asserts
neither disjointness nor a lesson sequence. The broader field inventory remains
in navigation collections, not an invented complete hierarchy.

Each collection names its field family; membership is derived from the existing
concept family assignment. Definitions are not copied into the taxonomy file.
Collection guiding questions are scope notes. Pilot top concepts are explicit;
unplaced field concepts are not automatically promoted to taxonomy roots.
The normal concept build validates and exports the taxonomy to SKOS Turtle and
a written draft reference. The field explorer renders the pilot hierarchy and
collections and opens the existing focused concept map for leaf selections.
External sources remain editorial references rather than student destinations.

These artifacts implement the initial sketch. Hierarchy review so far is AI
critique informed by published standards and curricula, not direct human expert
feedback or an observed learner study. Human review requests must identify a
recipient and include an accessible reviewed artifact rather than a localhost
link. No community endorsement, novelty, or exhaustive coverage is claimed.

### Student navigation

The explorer presents topics, a topic-scoped concept list, and concept detail.
Collections use a plain list; only explicitly classified branches have hierarchy
disclosures. Global search matches labels, aliases, and definitions, then opens
a result in its topic context. Book sections are the primary reading action.
Text connections are the default view, with a supplementary focused map tab.
The expanded map opens in its own tab and retains an explorer return link.

On phones the interface shows one navigation step at a time. The browser history
is authoritative: topic/concept transitions push state, while tab, zoom, and pan
changes replace it. Inline return controls use the matching prior history entry
when available. Search adds one entry when a search session begins. Restoration
retains the selected topic, concept, search, tab, and map viewport, with a saved
list focus target for returning to a long concept list. A measured header offset
keeps focus and sticky navigation clear of the site's responsive header.

Editorial status and interchange downloads remain available under About this
taxonomy / For editors. This presentation does not change taxonomy assertions,
concept definitions, or chapter ownership.
