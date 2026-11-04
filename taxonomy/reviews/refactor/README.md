# Taxonomy implementation refactor

The taxonomy code now has explicit module ownership under `binder/cli/core/taxonomy/`; see [Architecture](../../ARCHITECTURE.md). Existing binder commands and the public hierarchy imports remain available. Compatibility modules retain earlier Python imports without duplicating implementation.

The authored vocabulary, identifiers, schema, provenance history, and public formats were preserved. The download comparison reproduces 12 existing artifacts byte for byte using the complete field/book projection. Canonical source SHA-256: `929e9eddc790630af71b5076f11c32f350fc623d6cdf8361641ad51407d34c4f`. The initial comparison used an incomplete field projection and was corrected to use the same loader as the build; the saved receipt is the completed comparison.

Changes beyond file extraction: README inventory counts no longer serve as a schema-validation dependency; generated reports own live counts. New editorial receipts fingerprint the package as well as the historical entry point. Historical receipts keep their original interpretation. Editorial evidence, entry-term, impact, and draft-release checks are named helpers. Architecture tests check dependency cycles and compatibility; historical estimator and release-rejection checks are now exercised explicitly.

## Verification

The regression run passed 210 tests. [Test output](tests.log), [source/provenance check](check.log), [generated-asset freshness check](build-check.log), [export comparison](export-parity.json), and [coverage data](coverage.json) are retained here. The same test list is in the taxonomy CI workflow. Import-order and undefined-name checks pass. No website presentation files changed in this refactor, so earlier visual checks remain relevant; the refactor did not perform a new accessibility audit.

## Measured test coverage

These percentages were computed from coverage.py's raw statement and branch counts during the recorded test run. They describe exercised code, not semantic correctness or independent verification of historical results.

| Module | Statements exercised | Branches exercised |
|---|---|---|
| `__init__.py` | 100.0% | No branches |
| `editorial.py` | 94.4% | 91.1% |
| `exports.py` | 98.2% | 93.3% |
| `governance.py` | 1.8% | 0.5% |
| `hierarchy.py` | 93.0% | 88.0% |
| `history.py` | 57.2% | 45.3% |
| `legacy_policy.py` | 100.0% | No branches |
| `metrics.py` | 90.4% | 80.7% |
| `provenance.py` | 53.4% | 42.2% |
| `records.py` | 68.6% | 44.1% |
| `schema.py` | 87.2% | 74.6% |

The historical governance verifier remains large and has low execution coverage. Its source was separated without rewriting the recorded protocol. Broader end-to-end historical fixtures are needed before changing that machinery. Provenance and historical replay also have rejection paths that the current suite does not fully exercise. This refactor establishes ownership and regression protection; it does not claim exhaustive coverage or that every legacy function is small.

Coverage can be reproduced with `COVERAGE_FILE=/tmp/taxonomy.coverage python3 -m coverage run --branch --source=core.taxonomy -m pytest` followed by the taxonomy test paths listed in the CI workflow, then `COVERAGE_FILE=/tmp/taxonomy.coverage python3 -m coverage json -o /tmp/taxonomy-coverage.json`. Coverage is a development tool; it is not a new runtime dependency or CLI for the taxonomy.

No commits, pushes, or deployments were performed. The preexisting staged proposals were preserved.
