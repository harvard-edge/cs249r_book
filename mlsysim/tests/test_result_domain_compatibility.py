"""Domain ownership keeps legacy result imports and serialized identities intact."""

import ast
import importlib
import pickle
from pathlib import Path

import pytest

from mlsysim import Q_
from mlsysim.engine import result_types, results


def _defined_results():
    """Discover authored classes rather than maintaining a second domain catalog."""
    for path in sorted(Path(result_types.__file__).parent.glob("*.py")):
        module = importlib.import_module(f"mlsysim.engine.result_types.{path.stem}")
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef):
                yield module, node.name, getattr(module, node.name)


_RESULT_CLASSES = list(_defined_results())


@pytest.mark.parametrize("domain,name,model", _RESULT_CLASSES, ids=[row[1] for row in _RESULT_CLASSES])
def test_domain_and_legacy_imports_share_class_and_pickle_identity(domain, name, model):
    assert getattr(results, name) is model
    assert name in domain.__all__
    assert model.__module__ == "mlsysim.engine.results"
    assert model.__qualname__ == name
    assert model.__pydantic_complete__
    assert pickle.loads(pickle.dumps(model)) is model
    # No fabricated values: construction exercises defaults without validation.
    instance = model.model_construct()
    restored = pickle.loads(pickle.dumps(instance))
    assert type(restored) is model
    assert restored.__dict__ == instance.__dict__
    assert restored.model_fields_set == instance.model_fields_set


def test_every_legacy_wildcard_export_resolves_to_a_domain_definition():
    authored = {name: model for _, name, model in _RESULT_CLASSES}
    assert len(authored) == len(_RESULT_CLASSES), "A result must have one owning module"
    namespace = {}
    exec("from mlsysim.engine.results import *", namespace)
    assert {name for name in namespace if name != "__builtins__"} == set(results.__all__)
    for name in results.__all__:
        assert namespace[name] is authored[name]


def test_nested_result_annotation_validation_and_json_round_trip():
    result = results.DetectionShortfallResult(
        frame_period=Q_(10, "ms"),
        reference_frames=1,
        braking_travel=Q_(2, "m"),
        rows=[
            {
                "per_frame_detection": 0.8,
                "frames": 2,
                "extra_frames": 1,
                "extra_time": Q_(10, "ms"),
                "distance_lost": Q_(0.01, "m"),
                "impact_speed": Q_(0.1, "m/s"),
                "impact_energy": Q_(0.5, "J"),
            }
        ],
        within_braking=True,
    )
    assert type(result.rows[0]) is results.DetectionShortfallRow
    restored = results.DetectionShortfallResult.model_validate_json(result.model_dump_json())
    assert type(restored.rows[0]) is results.DetectionShortfallRow
    assert restored == result
    assert restored.reference_stop is None
    assert restored.rows[0].extra_time == Q_(10, "ms")
    assert pickle.loads(pickle.dumps(result)) == result
