"""Prediction-gated rendering must execute, including exact zero durations."""
import ast
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(('part', 'prediction'), [('C', 'within'), ('D', 'faster-fits'), ('E', 'strict')])
def test_parallelism_unlocked_part_renders_without_losing_tabs(part, prediction):
    path = Path(__file__).resolve().parents[1] / 'vol2' / 'lab_06_dist_train.py'
    spec = spec_from_file_location('parallelism_unlocked_regression', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    _, defaults = module.app.run()
    prediction_name = part.lower() + '_prediction'
    tree = ast.parse(path.read_text())
    cell = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and any(
        isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == prediction_name
        for target in statement.targets) for statement in node.body))
    returned = next(statement for statement in cell.body if isinstance(statement, ast.Return)).value
    names = [name.id for name in returned.elts]
    overrides = {name: defaults[name] for name in names}
    overrides[prediction_name] = SimpleNamespace(value=prediction)
    outputs, rendered = module.app.run(defs=overrides)
    assert outputs
    assert rendered['audit'].missing  # The lesson renders; unrelated work stays incomplete.
    if part == 'C':
        assert any(record['pipeline_bubble_time']['value'] == 0 for record in rendered['c_records'].values())
