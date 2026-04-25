# Lab Development Protocol

This public stub tracks the release-facing protocol checks that are enforced by
the labs test suite.

## Release Invariants

1. Every lab must be a valid Marimo app.
2. Browser labs must include the WASM bootstrap and relative `mlsysim` wheel path.
3. Prediction widgets must be structured controls, not free-text prompts.
4. Interactive widgets must be returned through Marimo dataflow.
5. Runtime-installed WASM packages must be imported after `micropip.install(...)`.
6. Labs should render tabbed parts plus a synthesis section.
7. Labs should save decisions to the Design Ledger where applicable.

Run `python3 -m pytest labs/tests/ -v` from the repository root for the current
automated checks.
