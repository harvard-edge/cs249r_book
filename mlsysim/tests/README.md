# Finding and running tests

Run commands below from the `mlsysim/` directory after installing the development
requirements described in [CONTRIBUTING.md](../CONTRIBUTING.md).

```sh
python -m pytest tests
```

Tests are named for the behavior they protect. Existing filenames remain stable
so contributor commands, CI selections, and downstream tooling continue to work.
A scenario often combines several domains; its test belongs with the behavior
being checked rather than the textbook volume that happens to use it.

| Question | Start here |
| --- | --- |
| Does a physical equation obey units and limiting cases? | `test_physics_bounds.py`, `test_quantity_formulas.py`, `test_robotics_physics.py`, `test_drivetrain_physics.py` |
| Do engine and solver calculations agree with independent accounting? | `test_engine.py`, `test_shared_accounting.py`, `test_release_accounting.py`, `test_accounting_input_domains.py` |
| Do named scenarios preserve their expected calculations? | `test_calculation_scenarios.py`, `test_scenario_contract_repairs.py`, `test_golden_regressions.py` |
| Are registry values, units, and provenance consistent? | `test_registry_loader_contract.py`, `test_registry_no_duplicate_specs.py`, `test_schema_units.py`, domain-specific `test_*_registry.py` files |
| Are equations, assumptions, and examples documented? | `test_modeling_documentation.py` |
| Can readers find the public API in the generated website? | `test_scenario_api_reference.py`, `test_public_api_guide.py` |
| Does a change affect published calculations or consumers? | `test_paper_integrity.py`, `test_downstream_impact.py`, `test_lego_unit_invariants.py` |

For a focused documentation check:

```sh
python -m pytest tests/test_modeling_documentation.py
python scripts/check_modeling_docs.py
```

The documentation checker discovers classes in `engine/scenario_definitions/`
and calculated result properties in `engine/result_types/` recursively. It reads
physical source definitions, including package initializers and base classes.
Compatibility imports in `scenario_types.py` and `results.py` do not duplicate
coverage; definitions still living in either file remain covered. Historical
`__module__` values used for serialization do not determine discovery.

For a focused numerical check, select the relevant behavior tests rather than
running only the documentation examples:

```sh
python -m pytest tests/test_physics_bounds.py tests/test_quantity_formulas.py
python -m pytest tests/test_calculation_scenarios.py tests/test_shared_accounting.py
```

Passing examples verifies that documented calls run as written. It does not
establish that an approximation fits measured hardware. Scientific validation,
release accounting, and downstream prose checks have separate responsibilities.
Shared hardware and workload fixtures live in `conftest.py`; pytest discovers
them automatically.
