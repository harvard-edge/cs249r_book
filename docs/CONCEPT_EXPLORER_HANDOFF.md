# Concept Explorer handoff

## Purpose and current state

This branch builds an MLSysBook concept explorer spanning all four volumes. The user wants a readable taxonomy, a full interactive concept map with labeled connections, and a repeatable authoring pipeline. Students should be able to follow an idea into related book material without navigating through many disconnected pages. The subject framework and the books' teaching inventories are distinct records.

Work is isolated in the `MLSysBook-concept-explorer` worktree on branch `concept-explorer`. This is a local review checkpoint, not a publication or merge into `dev`.

The current interface cleanup is implemented. The taxonomy and concept relationships remain editorial drafts. No claim of field completeness, comprehensive manuscript extraction, human expert consensus, or demonstrated student learning is justified.

## Update: taxonomy v0.2 and page redesign (2026-10-21)

This section supersedes counts and interface details below where they differ.

- **Frame.** The framework is now version `0.2-draft`: 5 groups (Foundations, Stack, Lifecycle, Qualities, System types), 20 areas, 101 subjects. The field vocabulary has 735 concepts and 1,531 relationships, each using one of 17 controlled relations. `docs/ML_SYSTEMS_FRAMEWORK.md` states the placement rules, the system-type admission test, the background tier, and the declared-gap policy.
- **How it was built.** Three research passes: an external survey of 33 sources (no comprehensive ML-systems body of knowledge was found), a four-volume book inventory, and a critique of the 0.1 rows. Then four authoring passes, one per group, and an adversarial red-team review whose P0 findings were applied: GPU programming model, recommendation and embedding systems, multimodal, aliases, identity-checked scope references, a background tier, and published known gaps. A follow-up pass closed all 16 P1 gaps with 38 concepts. The 12 P2 findings are now the declared `gaps` for v0.3.
- **Compiler.** `framework.py` accepts `0.2-draft`, validates an optional `groups` tier, resolves `book_chapters` codes (`v2/08`) to chapter titles and clean URLs, and rejects `[[identity]]` scope references that name no row or area. `concept_graph.py` now emits the flattened published chapter URL (`/volN/NN_chapter/…`); the old `/volN/volN/…` form is a redirect after publish. Vol III and IV pages are not live yet.
- **Page.** `_workspace.qmd`, `study.js`, and `study.css` were rewritten: a field map, area views, a concept panel, an A–Z index, lenses, and URL-hash state, in the neutral mlsysbook.ai style with dark mode. The interactive whole-graph canvas is retired from the page; `field-all.svg` remains a download. Old 0.1 row deep links fall back to the overview.
- **Concept map and glossary (later pass).** The relationship graph was rebuilt for mapping: every link classified into a controlled relation, 530 links added so no concept has fewer than two and the graph is connected (mean path about five links, down from seven), and an audit that read each link literally corrected or removed 76. After a review of how established taxonomies present themselves (MITRE ATT&CK/ATLAS, SFIA, ESCO, CS2023, MeSH) and of concept-map conventions, the page was rebuilt as a matrix home plus entry pages (area, subject, concept) with a sidebar tree, facts box, and tables, and a table glossary. Each concept page carries a neutral local concept map drawn with the vendored Cytoscape build. `field_graph.py` validates the relation vocabulary and rejects duplicate concept pairs.
- **Taxonomy release.** Groups, areas, and subjects have nested readable codes (`P4.3`). The Taxonomy tab shows the overview figure, a full collapsible outline with codes (filterable, printable to PDF), and a Download & cite section. `binder/cli/core/taxonomy_release.py` writes CSV, JSON, SKOS Turtle, GraphML, and BibTeX to `site/taxonomy/downloads/` (ignored by Git, regenerated each build). License is the book's CC BY-NC-SA pending a decision; no DOI has been minted.
- **Coverage.** `binder concepts coverage --check` scores recorded checklists in `shared/concepts/coverage/`. Design input (214 course, conference, and standards topics): 100% covered. First held-out measurement: MLSys 2025–2026 papers 67% covered by a concept and 97% homed in a subject; Volume I–II primary chapter concepts 85% and 99.5%; Volume III–IV 70% and 100%. After closing the field-level gaps those tests surfaced, they read 88%, 89%, and 83% and are now design inputs; the next held-out test is MLSys 2027. Remaining open topics are listed as partial in each checklist. Next planned use: StaffML questions re-expressed in taxonomy subjects and concepts (phased; see the session plan).
- **Review status.** All new concepts, placements, relationships, and chapter pointers are AI-authored drafts. Five ledger decisions are now stale because the vocabulary changed. All source URLs were link-checked: 232 load directly, all 34 DOIs match their cited titles in Crossref, and ISO, RAND, and ACM pages that block scripts were matched by identifier. One dead host was replaced with its DOI.

## Start the review here

1. Read this note, `docs/ML_SYSTEMS_FRAMEWORK.md`, and `docs/CONCEPT_COVERAGE_AUDIT.md`.
2. Inspect the canonical website entry at `site/taxonomy/index.qmd` and its shared body, `site/taxonomy/_workspace.qmd`.
3. Run the checks below and examine the generated reconciliation report before adding concepts or changing relationships.
4. Review the subject model and relationship semantics separately from the visual presentation. A valid build proves structural consistency, not conceptual completeness.

## Authoritative inputs

| Source | Responsibility |
|---|---|
| `books/vol*/*/*_concepts.yml` | Chapter-owned primary/secondary concepts, teaching claims, learner graphs, and cross-chapter explorer propositions |
| `shared/concepts/ml-systems-field.yml` | Independent field vocabulary, definitions, aliases, conditional typed relationships, and reading references |
| `shared/concepts/ml-systems-framework.yml` | Candidate subject areas and rows, boundaries, display labels, facets, and investigation proposals |
| `shared/concepts/mlsysbook-alignment.yml` | Explicit candidate links from field concepts to section-backed book concepts |
| `shared/concepts/ml-systems-taxonomy.yml` | Small explicit SKOS hierarchy pilot; not the entire subject framework |
| `shared/concepts/coverage-ledger.yml` | Authored reconciliation decisions, reviewer attribution, manuscript quotations, and dependency fingerprints |

The current field vocabulary has 175 concepts and 195 relationships. The organizing framework has 12 areas and 56 subject rows. There are 102 candidate passage links involving 86 distinct field concepts. These are inventory counts, not coverage scores. Empty subject rows are proposals awaiting an inventory, not proof that a book omits their subject.

Every current field concept has one candidate maintenance home. Secondary placements are explicit. Subject collections do not automatically imply an exclusive taxonomy, an `is-a` relationship, a prerequisite, or a prescribed reading sequence.

## Website implementation

- `/taxonomy/` serves the canonical Concept Explorer. It retains the main Quarto navbar, announcement, typography, theme switch, and footer. The explorer itself uses neutral slate across all volumes.
- The primary modes are **Taxonomy** and **Concept map**. Taxonomy has Cards and Tree layouts. Concept map defaults to every current field concept and authored relationship; Neighborhood is optional.
- Selection preserves the full graph, emphasizes directly connected nodes/edges, and opens an in-page reader. Locate on map frames the selected concept and its neighbors. Search, keyboard navigation, pan/zoom, and an explicit expanded window are supported.
- Book passages remain behind one disclosure. Conditions appear with relationship details. Draft and unassessed teaching-depth wording is intentional.
- `study.js` and `study.css` are the current controller and scoped styles; their names are historical. `_workspace.qmd` is the single authored interface body.
- `taxonomy-study.qmd` includes the same body to preserve existing deep links and `?expanded=1` URLs. It is not a separate visual variation to maintain.
- `volume-maps.qmd` preserves the chapter-owned volume-map viewer. **Binder extracts its raw HTML as the portable chapter/atlas template.** Do not substitute the field explorer. There is a guard and regression coverage for this distinction; fallback to `index.qmd` only supports older installations/fixtures.
- `field.qmd` is the older field browser retained for compatibility. Its volume-map link points to `volume-maps.html`.
- The explorer uses `page-layout: custom` and an explicit main landmark. Avoid introducing raw `aside` markup that makes Quarto inject nested column grids into the workspace. This previously squeezed the canvas.
- Generated static atlas, activity matrix, systems-perspective figures, CSV, and written references remain available as exports. They have not been integrated into book PDFs.

At Fit, the entire field graph is still too dense to read every label. Zoom and Locate make selected connections readable. A subject-clustered graph, aggregated connections, and semantic zoom remain substantive design work; this cleanup does not finish that problem. Mobile supports drag and zoom buttons; pinch zoom is not implemented. The mobile reader is stacked in the page rather than obscuring the map with a fixed overlay.

## Pipeline and generated files

`binder/cli/core/concept_graph.py` compiles chapter graphs and portable viewers. `field_graph.py` compiles the independent field model and related book passages. `framework.py`, `framework_figures.py`, and `taxonomy.py` generate the table, visual exports, and SKOS pilot. `concept_coverage.py` implements reconciliation accounting.

`site/scripts/build_concept_explorer.py` is the Quarto pre-render hook. The book filter and builder under `books/shared/` provide chapter integration. Site CI and Binder Docker definitions add Graphviz as a dependency; Docker images and remote CI have not been built in this worktree.

Do not hand-edit generated JSON, DOT, SVG, Turtle, portable HTML, chapter `concepts-generated/`, or `books/_build/concepts/` reports. They are intentionally ignored by Git. Commit the YAML, compiler, interface, and documentation sources. The vendored Cytoscape source and license are included for the chapter/volume viewer; the current field map uses generated SVG.

## Reconciliation status

The audit accounts for 801 chapter declarations: 505 primary and 296 secondary, across 71 chapter inventory files. Repeated ideas in separate chapters remain separate declarations.

The ledger currently contains 16 **AI proposals** for the primary declarations in fleet orchestration (`vol2/08`) and test-time compute (`vol3/03`). Their 39 quotations were checked against manuscript source. Nine proposals identify candidates for new field concepts; four identify broader correspondences and three narrower correspondences. These proposals have not automatically created new vocabulary entries or book links. Another 785 declarations remain unreviewed.

All current decisions retain `reviewer_kind: ai`; there are no current human decisions. **Do not relabel AI proposals as human review to make a gate pass.** Review attribution is recorded metadata, not authenticated identity. One allocation-control-plane proposal was revised after checking the broad definition of execution placement; its reviewer field records that additional check.

The audit distinguishes suggestions, current decisions, stale decisions, and orphans. Exact label matches and existing passage links never resolve an entry automatically. Fingerprints bind targeted decisions to their declaration, cited sections, and dependent field concepts. Untargeted boundary/new-concept proposals conservatively depend on the full semantic vocabulary. Renaming a chapter or exact concept label orphans its catalog-derived ID and requires explicit rebinding.

The human reconciliation gate covers selected primary declarations only. Secondary declarations remain inventoried. Passing that gate would still not certify manuscript extraction completeness, relationship correctness, teaching depth, or coverage of the field.

## Commands and verification

From the repository root:

```sh
python3 -m pip install -r site/taxonomy/requirements.txt
# Install Graphviz through the local package manager if `dot` is unavailable.
./binder/binder concepts check
./binder/binder concepts build
./binder/binder concepts build --check
./binder/binder concepts audit
./binder/binder concepts audit --vol2
./binder/binder concepts audit --check
```

`audit --check` currently exits nonzero by design: human reconciliation is unfinished. Do not enable that gate as a mandatory CI check yet or weaken it to manufacture completeness. The ordinary audit command succeeds while reporting unresolved work. Reports are generated at `books/_build/concepts/coverage-audit.json` and `.md`; volume-scoped reports receive a suffix.

Focused regression suite:

```sh
python3 -m pytest \
  binder/tests/test_concept_maps_v2.py \
  binder/tests/test_learner_concept_graph.py \
  binder/tests/test_field_concept_graph.py \
  binder/tests/test_framework.py \
  binder/tests/test_taxonomy.py \
  binder/tests/test_concept_coverage.py -q
```

Render/preview the main page from `site/`:

```sh
quarto render taxonomy/index.qmd --to html
quarto preview taxonomy/index.qmd
```

Also render `taxonomy/taxonomy-study.qmd`, `taxonomy/volume-maps.qmd`, and `taxonomy/field.qmd` when changing their integration. The shared stats pre-render hook can update `site/config/stats-cache.json` and can retain cached values when external services are unavailable. Do not commit an unrelated stats refresh or weaken TLS validation to render the explorer.

The pages were rendered locally and visually inspected at desktop, tablet, and phone widths, in light and dark modes. Search, two-tab keyboard navigation, selected concept/relationship reading, complete node/edge retention, legacy expanded deep links, and expanded-window navigation were checked. Mobile controls and horizontal overflow were inspected. This is not a formal accessibility audit or a student usability study. No complete book/PDF builds or remote CI validation were performed.

Claude Opus 5.5 contributed model, audit-code, and website-design critiques. Its latest follow-up screenshot review failed to connect and then timed out; do not describe the final visual pass as Opus-approved. The final browser checks were performed locally instead.

## Recommended next work

1. Review this checkpoint for correctness and maintainability before expanding its scope. Pay special attention to chapter template extraction, Quarto integration, and the distinct meanings of each registry.
2. Continue declaration-by-declaration reconciliation, beginning with the recorded AI proposals. Verify manuscript context and field definitions, then decide which candidate additions merit new concepts and relationships. A related passage is not proof of equivalence or teaching depth.
3. Audit the chapter inventories against the actual manuscripts. YAML declarations do not establish that every important taught concept was extracted.
4. Review existing relationships for semantic support. Many chapter explorer propositions were authored from teaching claims; quoted provenance does not prove that a predicate follows from the quotation. Field reading references are explicitly not entailment proofs. Negative/contrast relationship rendering also deserves review.
5. Compare the field vocabulary against bounded, pinned external curricula/frameworks with recorded topic dispositions. Never substitute lexical-match percentages or a large graph for a completeness argument.
6. Improve the full graph with stable subject clusters and readable zoom-dependent detail while accounting for every authored edge. Test the design with student navigation tasks. Keep the whole-map option, pan/zoom, neutral cross-volume identity, and book-focused reading flow.
7. Treat competency specifications and print integration as separate unfinished work. Neither can be inferred from a browsing taxonomy.

Do not merge, push, publish, claim completeness, or replace human review solely because this checkpoint builds successfully.
