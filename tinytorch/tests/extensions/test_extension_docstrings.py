"""Every extension documents itself in the same shape (template.py, FILL THIS OUT).

Readers look for "How it is triggered" in the same place in every file, so the
sections must all be present and in order. Added 2026-10-13 with the
extension docstring pass.
"""
import importlib
from pathlib import Path

import pytest

import tinytorch.extensions as ext

SECTIONS = (
    "What it is:",
    "Why it exists:",
    "How it is triggered:",
    "How it works",
    "What it leaves out:",
    "Where to go next:",
)
NOT_EXTENSIONS = {"template"}
# .py files only: a compiled libtinytorch_*.so sits beside them after a build
EXTENSIONS = sorted(p.stem for p in Path(ext.__file__).parent.glob("*.py")
                    if not p.stem.startswith("_") and p.stem not in NOT_EXTENSIONS)


@pytest.mark.parametrize("name", EXTENSIONS)
def test_extension_docstring_has_every_section_in_order(name):
    doc = importlib.import_module(f"tinytorch.extensions.{name}").__doc__ or ""
    positions = [doc.find(section) for section in SECTIONS]
    missing = [s for s, p in zip(SECTIONS, positions) if p < 0]
    assert not missing, f"{name}.py docstring is missing {missing}"
    assert positions == sorted(positions), f"{name}.py docstring sections are out of order"


def test_template_skeleton_lists_the_same_sections():
    doc = importlib.import_module("tinytorch.extensions.template").__doc__
    for section in SECTIONS:
        assert section in doc, f"template.py skeleton is missing {section!r}"
