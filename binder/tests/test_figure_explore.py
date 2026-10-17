"""Format and fail-closed contracts for opt-in interactive body figures."""
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
FILTER = ROOT / 'books/_extensions/mlsysbook-ext/figure-explore/figure-explore.lua'
pytestmark = pytest.mark.skipif(not shutil.which('quarto'), reason='Quarto required')


def render(tmp_path, target, figure='fig-tensor-memory-layout', lesson='tensor-memory-layout'):
    source = tmp_path / 'figure.qmd'
    enrollment = f'data-explore="{lesson}"' if lesson is not None else ''
    source.write_text(f'''---
filters:
  - {FILTER.as_posix()}
---

See @fig-tensor-memory-layout.

::: {{#{figure} {enrollment} fig-env="figure" fig-cap="**Tensor layout**: Original static figure."}}

```text
1 2 3
4 5 6
```

:::
''')
    result = subprocess.run(['quarto', 'render', str(source), '--to', target],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_html_packages_dependencies_without_replacing_static_body(tmp_path):
    render(tmp_path, 'html')
    text = (tmp_path / 'figure.html').read_text()
    assert 'data-mlsys-explore="tensor-memory-layout"' in text
    assert text.count('figure-explore.js') == 1
    assert text.count('tensor-memory-layout.js') == 1
    assert 'Original static figure.' in text
    assert text.count('<figcaption') == 1
    assert '1 2 3' in text and '4 5 6' in text
    assert list((tmp_path / 'figure_files').rglob('tensor-memory-layout.js'))


@pytest.mark.parametrize('figure,lesson', [
    ('fig-unrelated', 'tensor-memory-layout'),
    ('fig-tensor-memory-layout', 'not-registered'),
])
def test_unknown_lesson_or_wrong_figure_fails_closed(tmp_path, figure, lesson):
    result = render(tmp_path, 'html', figure=figure, lesson=lesson)
    text = (tmp_path / 'figure.html').read_text()
    assert 'unsupported lesson/figure pair' in result.stderr
    assert 'data-mlsys-explore=' not in text
    assert 'figure-explore.js' not in text
    assert 'Original static figure.' in text


def test_latex_keeps_static_figure_even_if_filter_is_explicitly_loaded(tmp_path):
    render(tmp_path, 'latex')
    text = (tmp_path / 'figure.tex').read_text()
    assert 'Original static figure.' in text
    assert '1 2 3' in text and '4 5 6' in text
    assert 'figure-explore' not in text
    assert 'Diagram' not in text and 'Explore' not in text


def test_epub_keeps_static_figure_even_if_filter_is_explicitly_loaded(tmp_path):
    render(tmp_path, 'epub')
    with zipfile.ZipFile(tmp_path / 'figure.epub') as archive:
        names = archive.namelist()
        text = '\n'.join(archive.read(n).decode() for n in names if n.endswith('.xhtml'))
    assert 'Original static figure.' in text
    assert '1 2 3' in text and '4 5 6' in text
    assert not any('figure-explore' in n or 'tensor-memory-layout.js' in n for n in names)
    assert 'data-mlsys-explore=' not in text


def test_unenrolled_figures_load_no_runtime(tmp_path):
    result = render(tmp_path, 'html', lesson=None)
    text = (tmp_path / 'figure.html').read_text()
    assert 'figure-explore.js' not in text
    assert 'data-mlsys-explore=' not in text
    assert 'unsupported lesson/figure pair' not in result.stderr
    assert 'Original static figure.' in text
