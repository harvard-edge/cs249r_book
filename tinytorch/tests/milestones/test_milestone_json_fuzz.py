"""
Property-based fuzz tests for milestone/progress JSON parsing.

Extends test_malformed_state_files.py's hand-picked corruption cases
(wrong top-level type, missing keys) with randomized JSON structures:
arbitrarily nested objects/arrays/scalars, and a "completed_modules"
key holding arbitrary garbage instead of a list of strings. The
function under test must never raise, only ever return a set (possibly
empty).

Requires the optional `fuzz` dependency group; skips cleanly if
hypothesis isn't installed.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

hypothesis = pytest.importorskip("hypothesis", reason="fuzz tests need the optional 'fuzz' dependency group")
from hypothesis import given, settings, strategies as st

from tito.commands.milestone import _load_completed_module_numbers


def _load_with_progress_content(content: str) -> set:
    """Write content to a fresh temp dir's .tito/progress.json, chdir
    there, and call the function under test. Uses tempfile directly
    (not pytest's tmp_path fixture) since hypothesis's @given doesn't
    mix cleanly with function-scoped pytest fixtures."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tito_dir = Path(tmp_dir) / ".tito"
        tito_dir.mkdir(parents=True, exist_ok=True)
        (tito_dir / "progress.json").write_text(content, encoding="utf-8")

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_dir)
            return _load_completed_module_numbers()
        finally:
            os.chdir(original_cwd)


# Any JSON-serializable value, recursively: scalars, lists, and dicts with
# string keys, nested up to a few levels deep.
json_value = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(max_size=20),
    lambda children: st.lists(children, max_size=5) | st.dictionaries(st.text(max_size=10), children, max_size=5),
    max_leaves=20,
)


@given(json_value)
@settings(max_examples=200)
def test_load_completed_modules_never_crashes_on_arbitrary_json(value):
    """Any syntactically valid JSON document, regardless of shape, must
    produce a set (possibly empty), never an exception."""
    result = _load_with_progress_content(json.dumps(value))

    assert isinstance(result, set)


@given(st.dictionaries(st.text(max_size=10), json_value, max_size=5), json_value)
@settings(max_examples=200)
def test_load_completed_modules_never_crashes_on_arbitrary_dict_shape(extra_fields, completed_modules_value):
    """A well-formed top-level object but with 'completed_modules' set to
    an arbitrary value (not a list of strings) must not crash."""
    progress = dict(extra_fields)
    progress["completed_modules"] = completed_modules_value

    result = _load_with_progress_content(json.dumps(progress))

    assert isinstance(result, set)


@given(st.text(max_size=500))
@settings(max_examples=200)
def test_load_completed_modules_never_crashes_on_arbitrary_text(garbage_text):
    """Arbitrary (likely not even valid JSON) text content, truncated
    writes, binary-ish garbage, must fall back cleanly."""
    result = _load_with_progress_content(garbage_text)

    assert isinstance(result, set)
