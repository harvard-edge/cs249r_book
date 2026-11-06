# Maintaining the taxonomy code and files

The vocabulary is the product. Its durable assets are concept identities, definitions, relationships, editorial decisions, and version history. The explorer, book tooling, and StaffML are consumers of those assets.

The [W3C SKOS Primer](https://www.w3.org/TR/skos-primer/) describes concepts, labels, relationships, and collections. The folder and module layout below is a project engineering decision, not a standards requirement.

There is one authored vocabulary and one formal schema. Existing `binder` commands remain the operational entry point. The taxonomy does not require a separate CLI, service, database, or installation.

## Where things belong

| Location | Owns | Who changes it |
|---|---|---|
| `taxonomy/taxonomy.yml` | Concepts, browsing homes, relationships, guides, and retired identifiers | Editors, through recorded reviews |
| `taxonomy/schema/` | Formal record definitions and controlled values | Maintainers when the data contract changes |
| `taxonomy/provenance.yml` | Replayable change history tied to content hashes | Maintenance tools |
| `taxonomy/mlsysbook-alignment.yml` | Connections to teaching passages | Editors; this is separate from concept meaning |
| `taxonomy/README.md`, `METHOD.md`, `CHANGELOG.md` | Entry point, editorial method, and release history | Maintainers |
| `taxonomy/reviews/` | Decisions, source evidence, and validation receipts | Reviewers and validation tools |
| `taxonomy/coverage/`, `agreement/` | Evaluation inputs with their declared collection history | Evaluation contributors |
| `taxonomy/method-study/` | Preserved historical study artifacts | Historical recovery and verification tools |
| `binder/cli/core/taxonomy/` | Taxonomy validation, maintenance, and export implementation | Developers |
| `binder/tools/audit/*taxonomy*.py` | Thin review, inventory, archive, and historical recovery commands | Developers |
| `binder/tests/test_taxonomy*`, `test_linkml_schema.py`, `test_framework.py`, `test_field_concept_graph.py` | Unit, regression, and integration tests | Developers alongside code changes |
| `taxonomy/study.js`, `study.css`, `_workspace.qmd` | Explorer presentation | Website maintainers |
| `site/scripts/build_concept_explorer.py` | Stage the public allow-list into the website | Website build tooling |
| `taxonomy/downloads/`, generated data/figures/reference files, `site/taxonomy/`, `site/_build/` | Rebuildable publication outputs | Build tools only |
| `staffml/` | Questions and an application consuming stable taxonomy identifiers | StaffML maintainers |

The canonical files use the short names `taxonomy.yml` and `provenance.yml`. Historical receipts retain the original filenames they recorded; public download URLs remain stable. The generated-file contract and ignore rules identify derived artifacts; generated outputs must never become a second authored vocabulary. Inventory counts belong in generated reports and exports, not duplicated assertions in README prose.

## Implementation ownership

`binder/cli/core/taxonomy/` is a Python package inside the existing application:

| Module | Responsibility |
|---|---|
| `schema.py` | Compile LinkML and check current record, graph, and guide constraints |
| `hierarchy.py` | Validate and export the explicit SKOS hierarchy slice |
| `editorial.py` | Validate and apply accepted changes to a copy of the vocabulary |
| `records.py` | Shared identifier/version syntax, deterministic JSON, and fingerprints |
| `provenance.py` | Verify the change chain and reconstruct provenance state |
| `exports.py` | Produce JSON, CSV, SKOS, GraphML, citation files, and StaffML labels |
| `history.py` | Replay historical campaigns and reconstruct their source ledger |
| `legacy_policy.py` | Pinned historical baselines and study-specific thresholds |
| `metrics.py` | Historical study estimators and their numerical checks |
| `governance.py` | Verify the historical reviewed-release protocol |

Historical rules are preserved for replay and remain distinct from the everyday editorial path. Their thresholds are project history, not ANSI/NISO, ISO, or W3C requirements. Extracting them does not certify the historical study or relax release approval.

`field_graph.py` and `framework.py` remain book/application adapters: they load and organize the vocabulary for existing binder consumers. The older `taxonomy_schema.py`, `taxonomy_editorial.py`, and `taxonomy_release.py` paths contain explicit compatibility imports. They own no duplicate implementation. New code imports the responsible package module directly. The existing `core.taxonomy` hierarchy imports also remain supported.

## Dependency direction

Record primitives and pinned policy sit at the bottom. History and metrics use those primitives. Historical governance uses history and metrics. Provenance uses these components to verify earlier decisions. Current schema validation invokes provenance verification where required. Editorial changes and export generation have separate responsibilities.

Package modules must not import their compatibility wrappers. Automated architecture tests reject those dependencies and cycles between package modules. Book loading adapters are called where integration requires them; the historical study does not acquire responsibility for editing live concepts.

The command scripts coordinate work: read a proposal, call library functions, validate the candidate, write recoverable files, and report the outcome. New rules belong in the library, not duplicated in commands, website JavaScript, or downstream applications.

## Safe maintenance

1. Edit vocabulary content through an accepted review; change code when the rules or representation change.
2. Preserve identifiers, exact history, and old receipt interpretation. New review receipts fingerprint the implementation package in addition to the historical compatibility entry point.
3. Run the regression suite listed in `.github/workflows/taxonomy-validate-dev.yml`. Add a test for a changed invariant or failure path, not merely for a helper's implementation.
4. Run `binder concepts check` and `binder concepts build --check`. Check StaffML after identity, home, or scope changes. Website changes additionally require rendered inspection. With the built site served locally, run `python3 site/scripts/check_taxonomy_ui.py <preview-url>` for browser interaction checks and `python3 site/scripts/check_concept_map_layout.py <preview-url>` for concept-map geometry and statement parity.
5. Compare generated outputs for a behavior-preserving refactor. Changing file organization should not alter vocabulary meaning or public formats.

Coverage is measured separately from pass/fail. A green suite does not imply every branch was exercised. Historical evaluation and release verification need their own fixtures and evidence; report their coverage separately from routine editorial operations.

The working tree's refactor validation receipts are under `taxonomy/reviews/refactor/`. Those receipts identify the tested implementation and the limits of the checks. Public release approval remains separate from code organization and test success.

## Reading conditional relationships in RDF

The Turtle export preserves relationship conditions in `rdf:Statement` records. Each record carries the relationship identifier (`dct:identifier`), endpoints, typed predicate, wording (`rdfs:label`), and application condition (`<https://mlsysbook.ai/taxonomy/scheme#condition>`). Conditions are human-readable text, not machine-executable rules.

Non-hierarchical typed relationships are no longer asserted as bare triples: doing so would discard their qualification for consumers that ignore statement metadata. Ordinary `skos:related` links remain available for browsing. Generic `specializes` links remain directly asserted and materialized as `skos:broader` because they must satisfy the all-instances hierarchy rule.

For example, retrieve the qualified reduction claims with:

```sparql
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX meta: <https://mlsysbook.ai/taxonomy/scheme#>
SELECT ?source ?target ?condition WHERE {
  ?statement a rdf:Statement ;
    rdf:subject ?source ;
    rdf:predicate <https://mlsysbook.ai/taxonomy/relation/reduces> ;
    rdf:object ?target ;
    meta:condition ?condition .
}
```

Consumers of earlier draft Turtle files that queried direct typed triples should migrate to these statement records. JSON and GraphML retain their existing relationship fields.

## Authoring and illustration tools

The authoring system and student interface serve different users. [LinkML](https://linkml.io/) (`taxonomy/schema/mlsys-taxonomy.linkml.yml`) defines the formal YAML metamodel for the taxonomy, mapping classes and slots to W3C `skos:`, `dct:`, and `owl:` URIs and compiling directly to JSON Schema (`mlsys-taxonomy.schema.json`). [VocBench](https://vocbench.uniroma2.it/) supports collaborative management of SKOS vocabularies and OWL ontologies. [Protégé](https://protege.stanford.edu/) supports OWL ontology development and reasoning.

For this repository, LinkML-validated YAML (`binder/cli/core/taxonomy_schema.py`) plus ontological graph checks in `binder concepts check` keeps the framework beside the book contracts and makes revisions inspectable in Git. The SKOS hierarchy comes only from `specializes` edges, audited with the Z39.19 all-and-some test; subjects stay collections and never become broader terms. A future editor integration should import/export through that LinkML and SKOS contract, rather than create a second independently edited truth.

The student illustrations use native HTML/SVG and generated, focused Graphviz relationship drawings. This supports readable typography, in-place selection, keyboard controls, deterministic static exports, and the same source data without a dependency on an external visualization service. Cytoscape remains available for graph interactions where it improves the learning task.
