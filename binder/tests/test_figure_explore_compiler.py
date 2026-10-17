"""Enrollment must never interpret braces inside chapter-owned attributes."""
import importlib.util
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('figure_compiler', ROOT / 'binder/tools/scripts/compile_figure_explore.py')
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)

@pytest.mark.parametrize('caption', [
    r'A $224{\times}224$ image',
    r'Cost $10^{22}$ and $\mathcal{O}(n)$',
    'Computed `{python} tile_count` values',
    'An ordinary caption',
])
def test_enrollment_leaves_every_original_byte_untouched(caption):
    source = '::: {#fig-demo fig-cap="' + caption + '" fig-alt="Original"}\n\nbody {x}\n:::\n'
    match = compiler.figure_match(source, 'fig-demo')
    marker = ' data-explore="guided-figure" data-explore-key="vol1:fig-demo"'
    enrolled = source[:match.end()] + marker + source[match.end():]
    assert enrolled.startswith('::: {#fig-demo' + marker + ' fig-cap=')
    assert compiler.normalized_source(enrolled) == source
    assert compiler.enrollment_attributes(enrolled, 'fig-demo') == marker


def test_structural_hash_ignores_generated_ids_but_preserves_geometry():
    original = b'<svg><metadata>date</metadata><path id="p123" d="M1 2"/><use href="#p123"/></svg>'
    changed_ids = b'<svg><metadata>other</metadata><path id="p456" d="M1 2"/><use href="#p456"/></svg>'
    assert compiler.structural_svg(original) == compiler.structural_svg(changed_ids)
    assert compiler.structural_svg(original) != compiler.structural_svg(original.replace(b'M1 2', b'M1 3'))
