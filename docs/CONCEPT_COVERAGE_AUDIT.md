# Chapter concept reconciliation

Run `./binder/binder concepts audit` to generate JSON and Markdown reports in `books/_build/concepts/coverage-audit.*`. Add `--vol1`, `--vol2`, `--vol3`, or `--vol4` to restrict the reported scope. Reports are derived artifacts; decisions belong in `taxonomy/coverage-ledger.yml`.

The audit inventories every declared primary and secondary concept in the chapter concept YAML files. It does not establish that those inventories exhaust the manuscripts, that relationships are correct, or that the field vocabulary is complete. Repeated concepts in separate chapters remain separate declarations.

Each inventory ID matches the curriculum catalog's goal ID: chapter directory plus an exact-label digest. Renaming a chapter or concept leaves the old decision orphaned. Reviewers must explicitly rebind it; the pipeline does not guess equivalence from similar labels.

## Decisions and suggestions

Label matches and existing related-passage links are suggestions only. They do not resolve an entry. A recorded decision includes:

- `inventory_id`, `chapter`, `label`, and `role` identifying the declaration.
- `disposition`: `mapped-to`, `narrower-than`, `broader-than`, `candidate-new`, or `out-of-scope`.
- `field_concepts`: existing target IDs for the first three dispositions, an empty list otherwise. Direction is from the chapter declaration to the field concept.
- `rationale`, `reviewer`, `reviewer_kind` (`human` or `ai`), and ISO `reviewed_on` date. These are recorded attributions, not authenticated identities.
- `evidence`: section IDs and verbatim manuscript source quotes. Preserve source markup, including emphasis and citations. Quotes must contain at least six source tokens to reject trivial strings; this threshold does not assess semantic support. A matching quote establishes provenance, not semantic correctness.
- `source_hash` and `field_hash` recording the dependencies reviewed.

`mapped-to` records an editorial correspondence; it is not a synonym assertion, proof of teaching depth, or SKOS equivalence. `candidate-new` proposes further vocabulary work and does not create a node automatically. `out-of-scope` requires an explicit reason.

The report distinguishes `unreviewed`, `current`, and `stale`. It reports current decisions by disposition and reviewer kind. AI proposals can be recorded as current against their evidence, but cannot satisfy the human review gate. A human must actually review a proposal before recording a human decision.

## Changes and review

The `decision_hashes` helper in `binder/cli/core/concept_coverage.py` computes fingerprints from the declaration, its claim, cited manuscript sections, and the field concepts referenced by a decision. An unrelated section edit or unrelated field addition does not invalidate a targeted mapping. Removed targets and changed or missing evidence reopen review. Decisions without field targets depend on the full semantic vocabulary inventory, so additions and definition changes conservatively reopen those boundary decisions.

The generated inventory exposes expected hashes for existing decisions. Updating a hash alone does not constitute review. The build never updates the ledger or accepts suggestions automatically. This first version retains one current decision per inventory ID; Git history preserves prior versions.

`./binder/binder concepts audit --check` fails when a primary declaration lacks a current human decision or a ledger entry is orphaned within the selected scope. Secondary declarations remain visible but do not gate this first review milestone. This check is intentionally not a completeness certificate and is not enabled as a CI requirement while reconciliation is underway.

The ledger schema is validated across all volumes even for a scoped report. Malformed records fail the audit globally; valid orphaned decisions affect only their selected volume. This separates ledger integrity from the scope of editorial work.

## Separate review tracks

This ledger covers chapter inventory reconciliation only. Manuscript extraction needs a separate record of candidate definitions, objectives, headings, and deliberate exclusions. Relationship review needs assertion-level passages, conditions, and reviewer decisions. Field coverage needs topic-by-topic comparison against explicitly pinned external frameworks, with a recorded disposition for each topic. None of these tracks can be replaced by a label-match percentage or a large graph.
