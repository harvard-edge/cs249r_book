# Readiness of the prepared editorial draft

Candidate: `0.4-draft`, canonical source SHA-256 `929e9eddc790630af71b5076f11c32f350fc623d6cdf8361641ad51407d34c4f`. This report concerns the working tree, not the preexisting Git index.

The local editorial preparation is complete. The candidate can be reviewed for publication as an explicitly provisional draft. It is not a certified standard or an independently validated community reference. Nothing has been committed, pushed, or deployed.

Implementation update: the subsequent behavior-preserving package refactor and its measured coverage are documented in [the refactor report](refactor/README.md). Its checks supersede the earlier implementation test run below; vocabulary content and compared downloads are unchanged.

## Completed preparation

- Resolve the documented compound boundaries: separate coalescing and shared-memory bank conflicts; distinguish graph tracing from compilation timing; distinguish risk acceptance criteria from ALARP.
- Withdraw unsupported safety relationships and qualify export/tracing dependencies. Sparse semantic graphs are reported instead of forcing unsupported edges; every concept retains a browsing home.
- Preserve search terms while removing automatic synonym assertions. Explicit source-backed decisions distinguish selected narrower terms and implementations from equivalent labels. Legacy terms with no decision remain marked as unassessed, not silently certified.
- Extend the public provenance-preserving workflow to additions, corrections, moves, explicit splits, mergers/retirements, entry-term decisions, and draft version transitions. Provide immutable local snapshots with version and content hashes. The reviewed-release governance gate remains separate and requires actual evidence.
- Complete the standards assessment at the level of inspected sources, explicitly identifying the unavailable normative ISO text. Prepare the independent expert-review, indexing, and navigation protocol without claiming participant results.

The [mechanical inventory](inventory.json) covers all 879 concepts and 2,010 relationships. It records review coverage, term decisions, lexical collisions, and sparse concepts. This is a vocabulary-wide consistency inventory, not an exhaustive semantic certification. The [assignment scope review](boundary-resolution/assignment-decisions.json) covers 48 affected StaffML records and retags 7; question answers and calculations were outside that screen.

## Validation receipts

| Check | Receipt and scope |
|---|---|
| Regression tests | [Test log](validation/tests.log): 205 passed. Covers schema, editorial replay, lifecycle guards, SKOS representation, recovery, and existing taxonomy behavior. |
| Source and provenance | [Check log](validation/check.log): full canonical/provenance validation passes. |
| Downstream consumers | [StaffML strict check](validation/vault.log) passes after assignment changes; schema enum/pattern synchronization also passes. |
| Historical judgments | [Coverage](validation/coverage.log) and [agreement](validation/agreement.log) recompute retained inputs. These logs do not verify their historical independence or constitute new human results. |
| Rendered site | [Render log](validation/render.log): taxonomy pages and downloads rebuilt; staging parity checked and internal evidence excluded from the public tree. |
| Concept maps | [Map log](validation/map-layout.log): every current concept map checked for label overlap and exact relationship wording/conditions. |
| Mobile and desktop | [Viewport checks](validation/visual-checks.json); screenshots of the new bank-conflict entry, isolated safety entry, and search-term display were inspected. No horizontal overflow in those views. This is not a full accessibility audit. |
| Frozen downloads | [Archive log](validation/archive.log): freshness check and local versioned snapshot; the archive manifest fingerprints its included source, provenance, methodology, schema, and downloads. |
| Exact files | [Candidate manifest](candidate-manifest.json) identifies the reviewed local source, tools, documentation, outputs, and receipts. |

The assignment changes altered usage counts in JSON/CSV downloads. The archive freshness guard caught this; the downloads were rebuilt before the final snapshot. Remote CI has not run. Original staged proposals remain intact; inspect the full working-tree diff before committing.

## Evidence still required for a reviewed release

Actual independent domain judgments and human indexing/navigation trials remain to be conducted using [the prepared protocol](human-review.md). Full normative ISO-text verification, broader assertion-level review, public archival hosting, live redirect checks, and deployment approval are not claimed complete. New equivalence claims and external mappings require their own source-backed decisions.

The supported standards wording is “informed by ANSI/NISO and ISO vocabulary practices, with a SKOS representation.” See [the standards assessment](standards-check.md), [current dispositions](dispositions.md), and [maintenance operations](lifecycle.md). Becoming a widely adopted reference will depend on evidence of usefulness, continuing maintenance, and community participation.
