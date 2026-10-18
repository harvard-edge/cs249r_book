"""Render actual TikZ to SVG: default seed, author override, and cache migration."""
import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
FILTER = ROOT / 'books/_extensions/pandoc-ext/diagram/diagram.lua'
pytestmark = pytest.mark.skipif(
    any(not shutil.which(tool) for tool in ('quarto', 'pdflatex', 'pdftocairo')),
    reason='Quarto and TeX/Poppler required',
)
BODY = r'''\begin{tikzpicture}
\foreach \x in {0,...,12} {
  \pgfmathsetmacro{\shade}{rnd*90+5}
  \fill[black!\shade] (\x,0) rectangle ++(1,1);
}
\end{tikzpicture}'''


def render(directory, body=BODY, cache=None, diagram_filter=FILTER):
    directory.mkdir(parents=True)
    cache_yaml = f'  cache: true\n  cache-dir: "{cache.as_posix()}"\n' if cache else ''
    source = directory / 'figure.md'
    source.write_text(f'''---
diagram:
{cache_yaml}  engine:
    tikz: {{}}
---

```{{.tikz}}
{body}
```
''')
    result = subprocess.run(
        ['quarto', 'pandoc', str(source), '--lua-filter', str(diagram_filter),
         '--to', 'html', '--extract-media', 'media', '-o', 'figure.html'],
        cwd=directory, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    files = list((directory / 'media').rglob('*.svg'))
    assert len(files) == 1, (files, result.stderr)
    return files[0].read_bytes()


def test_independent_uncached_tikz_renders_are_identical(tmp_path):
    assert render(tmp_path / 'first') == render(tmp_path / 'second')


def test_authored_seed_overrides_renderer_default(tmp_path):
    override = '\\pgfmathsetseed{27}\n' + BODY
    first = render(tmp_path / 'first', override)
    assert first == render(tmp_path / 'second', override)
    assert first != render(tmp_path / 'default')


def test_tikz_ignores_old_cache_then_reuses_versioned_cache(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    legacy = cache / (hashlib.sha1(BODY.encode()).hexdigest() + '.svg')
    legacy.write_bytes(b'<svg>old unseeded rendering</svg>')
    spaced_v1 = cache / (hashlib.sha1(('tikz-default-seed-1:v1\n' + BODY).encode()).hexdigest() + '.svg')
    spaced_v1.write_bytes(b'<svg>incorrect spacing from seed v1</svg>')
    first = render(tmp_path / 'first', cache=cache)
    expected = cache / (hashlib.sha1(('tikz-default-seed-1:v2\n' + BODY).encode()).hexdigest() + '.svg')
    assert expected.read_bytes() == first
    assert legacy.read_bytes() == b'<svg>old unseeded rendering</svg>'
    assert spaced_v1.read_bytes() == b'<svg>incorrect spacing from seed v1</svg>'
    # A valid marker proves the second pass reads the new entry rather than
    # silently regenerating the same image or accepting the legacy entry.
    cached = first.replace(b'<svg ', b'<svg data-cache-proof="reused" ', 1)
    expected.write_bytes(cached)
    assert render(tmp_path / 'second', cache=cache) == cached


def test_default_seed_introduces_no_spacing_or_nonrandom_art_changes(tmp_path):
    # TeX consumes an unprotected newline after the seed as horizontal space.
    # Compare the real SVG to the old template, including math and text glyphs.
    previous = tmp_path / 'old-diagram.lua'
    previous.write_text(FILTER.read_text().replace(
        r'\pgfmathsetseed{1}%' + '\n', ''
    ))
    body = r'''\begin{tikzpicture}
\node[draw, minimum width=4cm] (a) at (0,0) {$h_t = \sigma(W x_t + U h_{t-1})$};
\node[draw] (b) at (5,0) {Hidden state};
\draw[->] (a) -- (b);
\end{tikzpicture}'''
    assert render(tmp_path / 'old', body, diagram_filter=previous) == render(tmp_path / 'new', body)
