"""Exercise reviewed-art bindings through actual Quarto, including fail-closed drift."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[2]
FILTER = Path('books/shared/filters/reviewed_figure_assets.lua')
ART = Path('books/_extensions/mlsysbook-ext/figure-explore/canonical-art')
REGISTRY = ART.parent / 'figure-registry.json'
SOURCE = Path('books/vol1/08_training/08_training.qmd')
DIAGRAM = Path('books/config/shared/base/diagram.yml')
RENDERER = Path('books/_extensions/pandoc-ext/diagram/diagram.lua')
KEY = 'vol1:fig-fetching-optimized'
pytestmark = pytest.mark.skipif(
    any(not shutil.which(tool) for tool in ('quarto', 'lualatex', 'pdftocairo')),
    reason='Quarto, TeX, and Poppler required for the actual diagram pipeline',
)


@pytest.fixture
def project(tmp_path):
    paths = [FILTER, REGISTRY, SOURCE, DIAGRAM, RENDERER]
    paths += [p.relative_to(ROOT) for p in (ROOT / ART).rglob('*') if p.is_file()]
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    return tmp_path


def manifest(project):
    return json.loads((project / ART / 'manifest.json').read_text())


def save_manifest(project, value):
    (project / ART / 'manifest.json').write_text(json.dumps(value))


def render(project, *, mutation=None, override=None, source=SOURCE, target='html'):
    output = project / 'books/_build/html'
    filters = []
    command = ['quarto', 'render', str(project / source), '--no-execute', '--to', target,
               '--metadata-file', str(project / DIAGRAM)]
    if override:
        extra = project / 'override.yml'
        extra.write_text(override)
        command += ['--metadata-file', str(extra)]
    if mutation:
        extra = project / 'mutate.lua'
        extra.write_text(mutation)
        filters.append(str(extra))
    filters.append(str(project / FILTER))
    if target == 'html':
        filters.append(str(project / RENDERER))
    # Quarto user filters run before its resource extraction. Raw Pandoc
    # --lua-filter flags run too late to exercise Quarto's media-copy pipeline.
    config = project / 'books/_quarto.yml'
    config.write_text(json.dumps({'project': {'type': 'default'}, 'filters': filters}))
    command += ['--output-dir', str(output)]
    result = subprocess.run(command, cwd=project, text=True, capture_output=True)
    (project / 'render.log').write_text(result.stdout + result.stderr)
    return result, output


def alter_ast(code):
    return "local function mutate(d) if d.identifier=='fig-fetching-optimized' then " + code + ' end end;return {{Div=mutate,FloatRefTarget=mutate}}'


def test_quarto_emits_exact_reviewed_bytes(project):
    result, output = render(project)
    assert result.returncode == 0, result.stdout + result.stderr
    html = next(output.rglob('08_training.html')).read_text()
    for entry in manifest(project)['figures'].values():
        assert f'data-mlsys-reviewed-art-sha1="{entry["asset_sha1"]}"' in html
        candidates = list(output.rglob(entry['asset_sha1'] + '.svg'))
        assert len(candidates) == 1
        data = candidates[0].read_bytes()
        assert hashlib.sha1(data).hexdigest() == entry['asset_sha1']
        assert ET.fromstring(data).get('viewBox') == entry['view_box']


@pytest.mark.parametrize('control, message', [
    ('source', 'source changed'),
    ('body', 'TikZ body changed'),
    ('options', 'TikZ block options changed'),
    ('lesson_enrollment', 'figure lesson enrollment changed'),
    ('source_enrollment', 'source changed'),
    ('header_override', 'effective diagram configuration changed'),
    ('renderer', 'renderer/template/seed changed'),
    ('declared_input', 'declared input changed'),
    ('unknown_input', 'missing declared file'),
    ('missing_input_binding', 'missing shared diagram input binding'),
    ('missing_asset', 'missing declared file'),
    ('edited_asset', 'canonical artwork bytes changed'),
    ('geometry', 'canonical artwork geometry changed'),
    ('missing_block', 'expected exactly one TikZ block'),
    ('duplicate_wrapper', 'ambiguous figure wrapper'),
    ('missing_wrapper', 'missing figure wrapper'),
    ('missing_binding', 'missing canonical binding'),
    ('wrong_source', 'registry/provenance binding changed'),
    ('wrong_input', 'wrong owning source'),
])
def test_binding_drift_stops_render(project, control, message):
    data = manifest(project)
    entry = data['figures'][KEY]
    mutation = override = None
    source = SOURCE
    if control == 'source':
        with (project / SOURCE).open('a') as file:
            file.write('\nSource drift control.\n')
    elif control == 'body':
        mutation = alter_ast("d.content=pandoc.Div(d.content):walk{CodeBlock=function(b) b.text=b.text..'\\n% drift';return b end}.content;return d")
    elif control == 'options':
        mutation = alter_ast("d.content=pandoc.Div(d.content):walk{CodeBlock=function(b) b.attributes.width='99%';return b end}.content;return d")
    elif control == 'lesson_enrollment':
        mutation = alter_ast("d.attributes['data-explore']='unsupported-lesson';return d")
    elif control == 'source_enrollment':
        source_file = project / SOURCE
        source_file.write_text(source_file.read_text().replace(
            'data-explore="original-v1-fetching-optimized"', 'data-explore="unsupported-lesson"'))
    elif control == 'header_override':
        override = 'diagram:\n  engine:\n    tikz:\n      header-includes: ["\\\\usepackage{changed}"]\n'
    elif control == 'renderer':
        with (project / RENDERER).open('a') as file:
            file.write('\n-- Renderer contract drift\n')
    elif control == 'declared_input':
        with (project / DIAGRAM).open('a') as file:
            file.write('\n# Same effective config, changed declared input bytes\n')
    elif control == 'unknown_input':
        data['declared_inputs'].append({'path': 'books/missing-input.csv', 'sha1': '0' * 40})
    elif control == 'missing_input_binding':
        data['declared_inputs'] = []
    elif control == 'missing_asset':
        (project / ART / entry['asset']).unlink()
    elif control == 'edited_asset':
        with (project / ART / entry['asset']).open('ab') as file:
            file.write(b'\n<!-- changed bytes -->\n')
    elif control == 'geometry':
        entry['view_box'] = '0 0 1 1'
    elif control == 'missing_block':
        mutation = alter_ast('d.content=pandoc.Div(d.content):walk{CodeBlock=function(b) return {} end}.content;return d')
    elif control == 'duplicate_wrapper':
        mutation = ("local function duplicate(d) if d.identifier=='fig-mixed-precision' then "
                    "d.identifier='fig-fetching-optimized';"
                    "d.attributes['data-explore-key']='vol1:fig-fetching-optimized';return d end end;"
                    "return {{Div=duplicate,FloatRefTarget=duplicate}}")
    elif control == 'missing_wrapper':
        mutation = alter_ast('return {}')
    elif control == 'missing_binding':
        del data['figures'][KEY]
    elif control == 'wrong_source':
        entry['source'] = 'books/vol1/other/other.qmd'
    elif control == 'wrong_input':
        source = Path('books/vol1/other/other.qmd')
        (project / source).parent.mkdir(parents=True)
        shutil.copyfile(project / SOURCE, project / source)
    save_manifest(project, data)
    result, _ = render(project, mutation=mutation, override=override, source=source)
    assert result.returncode != 0, f'{control} silently passed'
    assert '[reviewed-figure-assets]' in result.stderr
    assert message in result.stderr


def test_non_html_keeps_tikz_without_reading_assets(project):
    entry = manifest(project)['figures'][KEY]
    (project / ART / entry['asset']).unlink()
    result, output = render(project, target='gfm')
    assert result.returncode == 0, result.stdout + result.stderr
    assert '\\begin{tikzpicture}' in next(output.rglob('08_training.md')).read_text()


def test_filter_registration_is_html_only_and_before_diagram():
    html = (ROOT / 'books/config/shared/html/filters.yml').read_text()
    assert html.index('shared/filters/reviewed_figure_assets.lua') < html.index('pandoc-ext/diagram')
    for target in ('pdf', 'epub'):
        assert 'reviewed_figure_assets' not in (ROOT / f'books/config/shared/{target}/filters.yml').read_text()
