# Implementation review evidence

This directory holds compact public review reports and optional acceptance records for mechanism scenes. Reports may bind inspected PNGs and motion captures by SHA256 without storing every image in the repository. Reports must identify the actual build, cases, profiles, states, motion samples, observed findings, and limits of the inspection. Simulated reviewer inspection is not student testing and establishes no learning gain.

The compiler grants no review status by default. After explicit acceptance, pass both `--build-root` and `--review-record`. The record uses the following exact schema:

- `schema`: integer `1`.
- `build_root`: the exact repository-relative clean four-volume HTML run directory.
- `registry_sha256`: SHA256 of the exact generated `figure-registry.json` bytes.
- `build_summary_sha256`: SHA256 of that run's `summary.json` bytes.
- `lessons`: map from an enrolled lesson key to its accepted review.

Each lesson review contains exactly `verdict` (`accepted`), `reviewer_type` (text), `limitations` (text), `coverage`, and `evidence`. Coverage contains `cases` (explicit base/variant IDs), `profiles` (explicit `desktop-light`, `desktop-dark`, `narrow-light`, or `narrow-dark` names), and nonempty `endpoints`, `motion`, and `controls` descriptions. Evidence is a nonempty array of `{ "path": "repository-relative report path", "sha256": "report SHA256" }` objects. Referenced reports must be inside this directory and available locally; symlinks outside the directory are rejected.

The compiler verifies the clean build's emitted runtime, bundles, and lesson/art markers before applying the record. Registry, build-summary, or report drift rejects acceptance. Only explicitly listed keys receive `implementation-review-passed`; all other keys remain pending. The catalog retains the declared coverage and limits. Use the same `--build-root` and `--review-record` options with `--check`. Plain compilation intentionally clears review acceptance and stale build routes. Binding a build or review does not modify manuscript enrollment, runtime source, or unchanged bundles.

No acceptance record is supplied by this README.
