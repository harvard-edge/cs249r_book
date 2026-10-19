"""Resolve checker inputs from actual chapter and full-volume HTML outputs."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / 'binder/tools/scripts/check_original_figures.py'


def load_checker():
    spec = importlib.util.spec_from_file_location('original_figure_checker', CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = Path('books/vol3/04_inference/04_inference.qmd')


def job(name='vol3-html', chapter=None, **overrides):
    return dict(job=name, chapter=chapter, format='html', volume='vol3', ok=True, **overrides)


def output(run, entry):
    path = run / entry['job'] / 'output' / SOURCE.relative_to('books').with_suffix('.html')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('<html>Actual requested chapter</html>')
    return path


def test_full_volume_html_resolves_actual_chapter(tmp_path):
    entry = job()
    expected = output(tmp_path, entry)
    assert load_checker().resolve_chapter_html(tmp_path, [entry], SOURCE, 3) == expected


@pytest.mark.parametrize('chapter_first', [True, False])
def test_owning_chapter_preferred_when_both_exist(tmp_path, chapter_first):
    chapter = job('vol3-html-04_inference', '04_inference')
    volume = job()
    expected = output(tmp_path, chapter)
    output(tmp_path, volume)
    entries = [chapter, volume] if chapter_first else [volume, chapter]
    assert load_checker().resolve_chapter_html(tmp_path, entries, SOURCE, 3) == expected


@pytest.mark.parametrize('chapter_missing', [True, False])
def test_unusable_chapter_falls_back_to_full_volume(tmp_path, chapter_missing):
    chapter = job('vol3-html-04_inference', '04_inference')
    if not chapter_missing:
        chapter['ok'] = False
        output(tmp_path, chapter)
    volume = job()
    expected = output(tmp_path, volume)
    assert load_checker().resolve_chapter_html(tmp_path, [chapter, volume], SOURCE, 3) == expected


@pytest.mark.parametrize('field,value', [
    ('ok', False), ('ok', 'false'), ('volume', 'vol2'),
    ('format', 'pdf'), ('format', 'epub'), ('chapter', '05_serving'),
])
def test_ineligible_job_rejected_even_with_html(tmp_path, field, value):
    entry = job()
    entry[field] = value
    output(tmp_path, entry)
    with pytest.raises(FileNotFoundError, match='No successful vol3'):
        load_checker().resolve_chapter_html(tmp_path, [entry], SOURCE, 3)


def test_missing_actual_html_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match='vol3/04_inference/04_inference.html'):
        load_checker().resolve_chapter_html(tmp_path, [job()], SOURCE, 3)


def test_import_has_no_cli_or_browser_dependency_side_effects():
    script = '''import builtins, importlib.util, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('playwright', 'yaml')):
        raise AssertionError('Browser/config imports must remain inside main')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['checker', '--deliberately-invalid-argument']
spec = importlib.util.spec_from_file_location('checker', sys.argv[0] if False else %r)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert callable(module.resolve_chapter_html)
''' % str(CHECKER)
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True)
