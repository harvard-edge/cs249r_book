"""Every fleet lab restores the saved narrative lens without saving progress."""
import ast
from pathlib import Path

import marimo as mo
import pytest

from mlsysim.labs.state import DesignLedger
from mlsysbook_labs.contexts import CONTEXT_OPTIONS, preferred_context

LABS = sorted((Path(__file__).resolve().parents[1] / 'vol2').glob('lab_*.py'))


@pytest.mark.parametrize('path', LABS, ids=lambda p: p.stem)
@pytest.mark.parametrize(('saved_track', 'expected'), [('cloud', 'cloud'), ('unrecognized-old-track', 'tinyml')])
def test_actual_track_cell_restores_saved_lens_and_rejects_unknown_default(path, saved_track, expected):
    initial = DesignLedger(volume='vol2')
    initial.save(track=saved_track)  # Explicit fixture setup; no chapter completion.
    ledger = DesignLedger(volume='vol2')
    assert ledger.get_track() == saved_track
    history = dict(ledger._state.history)
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and any(
        isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'track'
        for target in statement.targets) for statement in node.body))
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(path), 'exec'), namespace)
    controls = {'ledger': ledger, 'mo': mo, 'set_evidence': lambda _value: None,
                'set_saved_discussion': lambda _value: None,
                'CONTEXT_OPTIONS': CONTEXT_OPTIONS, 'preferred_context': preferred_context,
                'set_capture': lambda _value: None, 'set_status': lambda _value: None}
    result = namespace['_'](**{parameter.arg: controls[parameter.arg] for parameter in function.args.args})
    assert result[0].value == expected
    assert ledger._state.history == history
    assert ledger.get_track() == saved_track
