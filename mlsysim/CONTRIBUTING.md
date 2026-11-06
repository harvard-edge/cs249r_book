# Developing MLSysIM

Use Python 3.10 or newer. Work in an isolated environment so edits and optional
plotting dependencies do not alter another project's installation.

```bash
cd mlsysim
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,viz,labs]'
make check
```

For the Quarto site, also install `.[docs]` and Quarto, then run `make docs`.
The base wheel includes the CLI, registries, and analytical solvers. Plotting,
notebooks, and optimization backends are optional extras. Do not raise base
dependency floors without checking the browser/Pyodide consumers.

## Where code belongs

Start with the [Python API guide](docs/choosing-an-api.qmd) to distinguish public
entry points from their implementations. Keep caller imports stable while
refactoring domain code. The existing public-export tests check that
`mlsysim.solvers` exposes the canonical solver implementations.

| Location | Responsibility |
|---|---|
| `mlsysim/core/` | Units, validation, exceptions, and provenance types |
| `mlsysim/physics/` | Reusable equations and accounting conventions |
| Registry packages and their `data/` directories | Hardware/model specifications and sourced reference data |
| `mlsysim/engine/solvers/` | Domain solvers composing physics and registry values |
| `mlsysim/engine/scenario_definitions/` | Typed reusable scenarios, grouped by modeling domain |
| `mlsysim/engine/scenario_catalog/` | Named input configurations in the same domains |
| `mlsysim/engine/result_types/` | Result schemas with units and constraint traces |
| `mlsysim/engine/scenario_types.py`, `scenarios.py`, `results.py` | Compatibility imports for the domain implementations |
| `mlsysim/solvers.py` | Stable public solver imports |
| `mlsysim/cli/`, `viz/`, `labs/` | Presentation and interaction |
| `tests/` | Algebraic invariants, boundary cases, regression and integration checks |
| `scripts/` | Development-time generation and verification |

Keep shared accounting in one place. For example, the serving and batching
solvers share `_decode.py`, and the public training-memory APIs use the same
optimizer storage convention in `physics/memory.py`. Domain modules may be
split while retaining public imports; avoid mixing a module move with a
numerical model change.

Follow [one serving calculation through the code](docs/code-walkthrough.qmd)
before adding a new class. Scenario definitions, named configurations, and result
schemas share domain names, so related files can be found without reading a
large registry from start to finish. Their public facades preserve imports used
by textbooks and applications. The [test map](tests/README.md) identifies
contract, domain, integration, and release checks.

The [modeling and documentation conventions](docs/modeling-conventions.qmd) define
the standard NumPy docstring sections, source attribution, and publication workflow.
The engine is organized by modeling domain; all four textbook volumes consume it.

## Modeling contracts

A solver should state its unit of work, valid regime, exclusions, and whether
an input is a physical specification or an assumed efficiency. Keep quantities
attached through computation and convert units at presentation boundaries.

- Transformer training batches contain sequences: sequence length scales work
  and activation storage. Generic inference evaluates one forward step.
- Serving decode bounds both compute and memory time. `efficiency` changes
  compute capacity only; peak memory bandwidth is a separate assumption.
- Speculative decoding holds both models and both KV caches. Disaggregated
  serving checks the prefill and decode devices separately.
- The memory footprint excludes engine workspaces and allocator reserve unless
  a solver explicitly includes them. A feasible result is not a deployment guarantee.
- Generic roofline activation traffic remains a declared heuristic, not an
  architecture-specific trace. TDP-based energy is an estimate, not meter data.
- Reject invalid and non-finite inputs at the public boundary. An infinite
  result can still represent an unstable queue; it is not a valid input rate.
  Explicit mathematical limits, such as infinite local speedup in Amdahl’s law,
  are documented exceptions.

## Tests and documentation

```bash
make lint                    # package and development scripts
make test                    # all package tests
make calibration-check       # generated report agrees with executable anchors
make modeling-docs-check     # scoped formatting, NumPy contracts, sources, examples
make build                   # wheel and source distribution
python -m twine check dist/*
```

For a formula change, first add a test from the physical/accounting contract:
conservation of bytes, monotonic work, a capacity boundary, or agreement between
public APIs with identical assumptions. A golden value alone cannot establish
correctness. Keep calibration, sanity ranges, and external validation distinct;
see [the calibration report](docs/empirical-calibration.md).

## Downstream impact

Before merging a numerical change, record before/after outputs for affected
scenarios and identify their textbook, lab, paper, and slide consumers. Review
both the computed values and the explanations that depend on them. Do not
silently update prose or repin golden tests to make a changed result pass.
Prose corrections are a separate editorial change.

From the repository root, the following are useful integration checks:

```bash
PYTHONPATH=mlsysim:labs python -m pytest -c labs/pytest.ini \
  labs/tests/test_engine.py labs/tests/test_rationale.py labs/tests/test_registry_refs.py
PYTHONPATH=mlsysim:binder python -m pytest \
  binder/tests/test_mlsysim_registry_coverage.py binder/tests/test_registry.py
PYTHONPATH=mlsysim python mlsysim/paper/scripts/generate_paper_values.py --check
PYTHONPATH=mlsysim python mlsysim/paper/scripts/validate_anchors.py
```

Use `scripts/downstream_impact.py snapshot` and `compare` to compare the same
textbook and paper consumers against baseline and candidate package versions.
The [release runbook](RELEASE.md#compare-downstream-numerical-outputs) gives the
commands, review statuses, and limits. Raw changes are reported even when rounded
prose is unchanged; associated explanations require separate editorial review.

These do not replace executing the affected textbook cells or checking rendered
figures and browser labs. Review their output before proceeding with a release.
