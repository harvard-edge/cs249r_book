"""Execute modeling documentation and exercise the validation failure paths."""

import inspect
import runpy
from pathlib import Path

import pytest

CHECKER = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "check_modeling_docs.py"))
CONFIG = CHECKER["configuration"]()


@pytest.mark.parametrize("name", CONFIG["objects"])
def test_modeling_contract(name):
    errors = CHECKER["check_object"](name, CONFIG["numpydoc_ignore"], CHECKER["bibliography_keys"]())
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize(
    "old,new,expected",
    [
        ("n_params : Quantity or int", "wrong_name : Quantity or int", "PR01"),
        ("Equation:", "Calculation:", "MODEL01"),
        ("Assumptions:", "Conditions:", "MODEL01"),
        ("Excludes:", "Omissions:", "MODEL01"),
        ("bibliography key: kaplan2020scaling", "bibliography key: missing_source", "MODEL04"),
        ("second == first * 2", "second == first * 3", "MODEL06"),
    ],
)
def test_checker_detects_broken_contract(monkeypatch, old, new, expected):
    name = "mlsysim.physics.transformer.calc_transformer_training_flops"
    obj = CHECKER["resolve"](name)
    doc = inspect.getdoc(obj)
    assert old in doc
    monkeypatch.setattr(obj, "__doc__", doc.replace(old, new))
    keys = {"kaplan2020scaling", "hoffmann2022chinchilla"}
    errors = CHECKER["check_object"](name, CONFIG["numpydoc_ignore"], keys)
    assert any(expected in error for error in errors), errors


def test_stale_target_fails():
    with pytest.raises(AttributeError):
        CHECKER["resolve"]("mlsysim.physics.transformer.nonexistent_equation")


@pytest.mark.parametrize("name,methods", CONFIG["scenarios"].items())
def test_scenario_calculation_contract(name, methods):
    errors = CHECKER["check_scenario"](name, methods, CHECKER["bibliography_keys"]())
    assert not errors, "\n".join(errors)


def test_new_calculations_are_discovered_without_allowlist(tmp_path):
    physics = tmp_path / "mlsysim/physics"
    solvers = tmp_path / "mlsysim/engine/solvers"
    physics.mkdir(parents=True)
    solvers.mkdir(parents=True)
    (physics / "future.py").write_text("def calc_new(x):\n    return x * 2\n")
    (solvers / "future.py").write_text("class NewModel:\n    def solve(self, x):\n        return x * 2\n")
    names = CHECKER["discover_contracts"](tmp_path)
    assert names == [
        "mlsysim.engine.solvers.future.NewModel.solve",
        "mlsysim.physics.future.calc_new",
    ]


def test_scenario_discovery_includes_new_properties(tmp_path):
    path = tmp_path / "mlsysim/engine/scenario_types.py"
    path.parent.mkdir(parents=True)
    path.write_text("class NewScenario:\n    @property\n    def amount(self):\n        return 2\n")
    assert CHECKER["discover_scenarios"](tmp_path) == {"mlsysim.engine.scenario_types.NewScenario": ["amount"]}


def test_original_derivation_does_not_need_an_unrelated_citation():
    source = "Derivation: divide transferred bytes by bytes per second to obtain seconds."
    assert CHECKER["check_references"](source, set()) == []
    assert CHECKER["check_references"]("Derivation: algebra.", set())
    assert CHECKER["check_references"]("This calculation follows first principles.", set())


def test_hyphenated_citation_key_is_checked_completely():
    source = "https://example.org/paper (bibliography key: author-year)."
    assert CHECKER["check_references"](source, {"author-year"}) == []
    assert any("MODEL04" in e for e in CHECKER["check_references"](source, {"author"}))


def test_checker_rejects_missing_sources(monkeypatch):
    name = "mlsysim.physics.power.calc_rail_current_step"
    obj = CHECKER["resolve"](name)
    doc = inspect.getdoc(obj)
    references_start = doc.index("References\n")
    examples_start = doc.index("Examples\n")
    monkeypatch.setattr(obj, "__doc__", doc[:references_start] + doc[examples_start:])
    errors = CHECKER["check_object"](name, CONFIG["numpydoc_ignore"], set())
    assert any("MODEL02" in e for e in errors)


def test_scenario_property_cannot_lose_its_trace(monkeypatch):
    name = "mlsysim.engine.scenario_types.DecodeScenario"
    obj = CHECKER["resolve"](name)
    method = obj.result.fget
    monkeypatch.setattr(method, "__doc__", "Return the modeled result.")
    errors = CHECKER["check_scenario"](name, ["result"], CHECKER["bibliography_keys"]())
    assert any("SCENARIO03" in e for e in errors)


def test_every_implementation_reference_is_resolved():
    refs = (
        "Implementation references: ``mlsysim.physics.transformer.calc_transformer_training_flops``, "
        "``mlsysim.physics.transformer.missing_formula``."
    )
    errors = CHECKER["check_references"](refs, set())
    assert any("missing_formula" in error and "MODEL07" in error for error in errors)


def test_scenario_trace_rejects_stale_qualified_target(monkeypatch):
    name = "mlsysim.engine.scenario_types.DecodeScenario"
    obj = CHECKER["resolve"](name)
    monkeypatch.setattr(
        obj.result.fget,
        "__doc__",
        "Return latency in seconds. Calculation: ``mlsysim.solvers.MissingModel.solve``.",
    )
    errors = CHECKER["check_scenario"](name, ["result"], CHECKER["bibliography_keys"]())
    assert any("SCENARIO04" in error for error in errors)


@pytest.mark.parametrize("name", CONFIG["api_objects"])
def test_composition_and_conversion_api_contract(name):
    errors = CHECKER["check_api_object"](name, CONFIG["numpydoc_ignore"])
    assert not errors, "\n".join(errors)


def test_equation_markup_preserves_multiplication():
    assert CHECKER["check_math_markup"]("Equation: time = work*rate")
    assert CHECKER["check_math_markup"]("Equation: time = ``work*rate``") == []
    assert CHECKER["check_math_markup"]("Equation: $work*rate$") == []
    assert CHECKER["check_math_markup"]("Examples\n--------\n>>> work*rate") == []


def test_scenario_discovery_follows_domain_sources_not_compatibility_identity(tmp_path):
    engine = tmp_path / "mlsysim/engine"
    domains = engine / "scenario_definitions"
    (domains / "nested").mkdir(parents=True)
    (engine / "scenario_types.py").write_text("from .scenario_definitions.nested.serving import DecodeScenario\n")
    (domains / "__init__.py").write_text("from .nested.serving import DecodeScenario\n")
    (domains / "base.py").write_text("class BaseScenario:\n    def variant(self):\n        return self\n")
    (domains / "nested/serving.py").write_text(
        "class DecodeScenario:\n"
        '    __module__ = "mlsysim.engine.scenario_types"\n'
        "    @property\n"
        "    def latency(self):\n"
        "        return 1\n"
    )
    assert CHECKER["discover_scenarios"](tmp_path) == {
        "mlsysim.engine.scenario_definitions.base.BaseScenario": ["variant"],
        "mlsysim.engine.scenario_definitions.nested.serving.DecodeScenario": ["latency"],
    }


def test_result_contract_discovery_includes_nested_and_initializer_definitions(tmp_path):
    engine = tmp_path / "mlsysim/engine"
    domains = engine / "result_types"
    (domains / "nested").mkdir(parents=True)
    (engine / "results.py").write_text("from .result_types.nested.serving import ServingResult\n")
    (domains / "__init__.py").write_text("class BaseResult:\n    @property\n    def ratio(self):\n        return 1\n")
    (domains / "nested/serving.py").write_text(
        "class ServingResult:\n"
        '    __module__ = "mlsysim.engine.results"\n'
        "    @property\n"
        "    def latency(self):\n"
        "        return 1\n"
        "    def summary(self):\n"
        '        return "presentation"\n'
    )
    assert CHECKER["discover_contracts"](tmp_path) == [
        "mlsysim.engine.result_types.BaseResult.ratio",
        "mlsysim.engine.result_types.nested.serving.ServingResult.latency",
    ]


def test_composition_scenario_keeps_ordinary_api_contract_after_move(tmp_path):
    engine = tmp_path / "mlsysim/engine"
    domains = engine / "scenario_definitions"
    domains.mkdir(parents=True)
    (engine / "scenarios.py").write_text("from .scenario_definitions.base import Scenario\n")
    (domains / "base.py").write_text("class Scenario:\n    pass\nclass BaseScenario:\n    pass\n")
    assert CHECKER["discover_scenarios"](tmp_path) == {"mlsysim.engine.scenario_definitions.base.BaseScenario": []}
    assert CHECKER["discover_api_contracts"](tmp_path) == [
        "mlsysim.engine.scenario_definitions.base.Scenario.evaluate",
        "mlsysim.engine.scenario_definitions.base.Scenario.is_distributed",
        "mlsysim.engine.scenario_definitions.base.Scenario.validate",
    ]
