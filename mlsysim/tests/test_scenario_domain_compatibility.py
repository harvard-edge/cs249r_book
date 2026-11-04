"""Public identity and construction dependencies survive domain organization."""

import ast
import importlib
import inspect
from pathlib import Path
import pickle

from pydantic import BaseModel

from mlsysim import Scenarios
from mlsysim.engine import scenario_types
from mlsysim.engine.scenarios import Scenario


ENGINE = Path(scenario_types.__file__).parent


def _modules(package):
    for path in sorted((ENGINE / package).rglob("*.py")):
        parts = path.relative_to(ENGINE).with_suffix("").parts
        yield path, "mlsysim.engine." + ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def test_every_authored_scenario_keeps_legacy_import_and_pickle_identity():
    """Check definitions directly, rather than trusting facade enumeration."""
    for path, module_name in _modules("scenario_definitions"):
        module = importlib.import_module(module_name)
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.ClassDef):
                continue
            cls = getattr(module, node.name)
            legacy = importlib.import_module(cls.__module__)
            assert cls.__module__ == (
                "mlsysim.engine.scenarios" if node.name == "Scenario" else "mlsysim.engine.scenario_types"
            )
            assert getattr(legacy, node.name) is cls
            assert pickle.loads(pickle.dumps(cls)) is cls
            assert issubclass(cls, BaseModel)


def test_catalog_objects_are_the_public_registry_objects():
    """Each case has one construction site and the facade preserves that object."""
    authored = {}
    for path, module_name in _modules("scenario_catalog"):
        module = importlib.import_module(module_name)
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name) or target.id.startswith("_"):
                    continue
                obj = getattr(module, target.id)
                if not isinstance(obj, BaseModel):
                    continue
                assert target.id not in authored, f"Duplicate construction of {target.id}"
                authored[target.id] = obj
                assert getattr(Scenarios, target.id) is obj
    registry = {name: value for name, value in vars(Scenarios).items() if not name.startswith("_") and name not in Scenarios.__registry_list_exclude__}
    assert registry.keys() == authored.keys()
    expected = []
    for name in sorted(authored):
        obj = authored[name]
        if not any(obj is existing for existing in expected):
            expected.append(obj)
    assert all(a is b for a, b in zip(Scenarios.list(), expected, strict=True))


def test_catalog_import_dependencies_are_acyclic():
    """Topic modules must compose cases without circular or runtime imports."""
    modules = dict((name, path) for path, name in _modules("scenario_catalog"))
    edges = {name: set() for name in modules}
    for name, path in modules.items():
        tree = ast.parse(path.read_text())
        package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"exec", "eval", "__import__"}
            if isinstance(node, ast.ImportFrom) and node.level:
                target = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                if target in modules:
                    edges[name].add(target)
    done = set()

    def visit(name, path=()):
        assert name not in path, f"Circular scenario construction: {path + (name,)}"
        if name in done:
            return
        for target in sorted(edges[name]):
            visit(target, path + (name,))
        done.add(name)

    for name in sorted(modules):
        visit(name)


def test_scenario_variant_and_nested_input_identity_remain_shallow():
    case = Scenarios.Llama3_70B_DecodeH100
    changed = case.variant(batch_size=3)
    assert changed is not case
    assert changed.batch_size == 3
    assert changed.workload is case.workload
    assert changed.hardware is case.hardware
    assert pickle.loads(pickle.dumps(case)).model_dump() == case.model_dump()
    assert Scenario.__module__ == "mlsysim.engine.scenarios"
    assert inspect.getmodule(type(case)) is scenario_types


def test_narrative_deployment_briefs_do_not_enter_numerical_catalog():
    """Narrative metadata remains public without posing as an executable scenario."""
    from mlsysim.engine.deployment_journeys import DeploymentJourneyCatalog

    assert Scenarios.__registry_list_exclude__ == frozenset({"DeploymentJourneys"})
    assert isinstance(Scenarios.DeploymentJourneys, DeploymentJourneyCatalog)
    assert all(value is not Scenarios.DeploymentJourneys for value in Scenarios.list())
    assert Scenarios.ControlledTaskFactors in Scenarios.list()
    assert Scenarios.PlacementStudy in Scenarios.list()
