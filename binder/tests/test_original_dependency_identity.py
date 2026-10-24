"""Original-module cache URLs must track the module bytes, not only shared assets."""
from hashlib import sha1
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / 'books/_extensions/mlsysbook-ext/figure-explore'
pytestmark = pytest.mark.skipif(not shutil.which('quarto'), reason='Quarto required')


def fixture(tmp_path):
    ext = tmp_path / 'books/_extensions/mlsysbook-ext/figure-explore'
    ext.mkdir(parents=True)
    for name in ('figure-explore.lua', 'figure-views.js', 'figure-explore.js',
                 'figure-views.css', 'original-player.js', 'original-player.css'):
        shutil.copyfile(EXT / name, ext / name)
    (ext / 'original-cache.js').write_text('window.originalCacheRevision=1;')
    (ext / 'original-cache.css').write_text('.original-cache { color: blue; }')
    source = tmp_path / 'books/vol1/cache/cache.qmd'
    source.parent.mkdir(parents=True)
    source.write_text(f'''---
figure-explore:
  movement-ms: 1800
  reading-pause-ms: 5000
filters:
  - {ext / 'figure-explore.lua'}
---

::: {{#fig-cache data-explore="original-cache" data-explore-key="vol1:fig-cache" fig-cap="Original static caption."}}

Static original body.

:::
''')
    normalized = re.sub(r'\s+data-explore(?:-key)?="[^"]*"', '', source.read_text())
    entry = dict(name='original-cache', figure_id='fig-cache', volume=1,
                 source='books/vol1/cache/cache.qmd', source_sha1=sha1(normalized.encode()).hexdigest(),
                 script='original-cache.js', stylesheet='original-cache.css', shared_player=True,
                 art_sha1='a' * 40, art_kind='svg')
    return ext, source, entry


def write_registry(ext, entry):
    (ext / 'figure-registry.json').write_text(json.dumps({'vol1:fig-cache': entry}))


def runtime_hash(ext):
    return sha1(b'\0'.join((ext / name).read_bytes() for name in (
        'original-cache.js', 'original-cache.css', 'original-player.js', 'original-player.css'))).hexdigest()


def render(source):
    result = subprocess.run(['quarto', 'render', str(source), '--to', 'html'],
                            cwd=source.parent, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return source.with_suffix('.html').read_text(), result.stderr


def module_url(html):
    matches = re.findall(r'<script src="([^"]*/original-cache\.js)"', html)
    assert len(matches) == 1
    return matches[0]


def test_module_byte_change_changes_emitted_url_without_shared_changes(tmp_path):
    ext, source, entry = fixture(tmp_path)
    shared = {name: (ext / name).read_bytes() for name in (
        'original-player.js', 'original-player.css', 'figure-explore.js', 'figure-views.js', 'figure-views.css')}
    entry['runtime_sha1'] = runtime_hash(ext)
    write_registry(ext, entry)
    first, _ = render(source)
    first_url = module_url(first)
    assert '0.2.0-' + entry['runtime_sha1'][:12] in first_url
    assert (source.parent / first_url).read_bytes() == (ext / 'original-cache.js').read_bytes()
    (ext / 'original-cache.js').write_text('window.originalCacheRevision=2;')
    entry['runtime_sha1'] = runtime_hash(ext)
    write_registry(ext, entry)
    second, _ = render(source)
    second_url = module_url(second)
    assert second_url != first_url
    assert '0.2.0-' + entry['runtime_sha1'][:12] in second_url
    assert (source.parent / second_url).read_bytes() == (ext / 'original-cache.js').read_bytes()
    assert all((ext / name).read_bytes() == content for name, content in shared.items())
    assert 'Static original body.' in second and 'Original static caption.' in second


@pytest.mark.parametrize('digest', [None, 'unversioned', 'g' * 40])
def test_original_without_valid_runtime_identity_retains_static(tmp_path, digest):
    ext, source, entry = fixture(tmp_path)
    if digest is not None:
        entry['runtime_sha1'] = digest
    write_registry(ext, entry)
    html, stderr = render(source)
    assert 'missing or invalid original runtime identity' in stderr
    assert 'data-mlsys-explore=' not in html
    assert 'original-cache.js' not in html
    assert 'Static original body.' in html and 'Original static caption.' in html


@pytest.mark.parametrize('name,hashed', [('convolution', True), ('guided-figure', False)])
def test_nonoriginal_specialized_hashes_but_guided_version_is_unchanged(tmp_path, name, hashed):
    ext, source, entry = fixture(tmp_path)
    source.write_text(source.read_text().replace('data-explore="original-cache"', f'data-explore="{name}"'))
    normalized = re.sub(r'\s+data-explore(?:-key)?="[^"]*"', '', source.read_text())
    entry['source_sha1'] = sha1(normalized.encode()).hexdigest()
    entry['name'] = name
    entry['runtime_sha1'] = runtime_hash(ext)
    (ext / 'guided-lessons-vol1.js').write_text('// Legacy guided data fixture')
    write_registry(ext, entry)
    html, _ = render(source)
    url = module_url(html)
    if hashed:
        assert '0.2.0-' + entry['runtime_sha1'][:12] in url
    else:
        assert 'mlsys-explore-guided-figure-vol1-0.2.0/' in url
        assert entry['runtime_sha1'][:12] not in url


def test_bundled_version_formula_is_unchanged(tmp_path):
    ext, source, entry = fixture(tmp_path)
    bundle = ext / 'values.js'
    bundle.write_text('window.cacheValues=[1,2];')
    entry.update(runtime_sha1=runtime_hash(ext), bundle=bundle.name,
                 bundle_sha1=sha1(bundle.read_bytes()).hexdigest(), lesson_sha1='a' * 40)
    write_registry(ext, entry)
    html, _ = render(source)
    assert '0.3.0-' + entry['runtime_sha1'][:12] + '-' + entry['bundle_sha1'][:12] in module_url(html)
