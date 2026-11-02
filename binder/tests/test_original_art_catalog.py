"""Catalog review evidence and links require actual owning chapter HTML."""
from hashlib import sha1
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('original_art_catalog', ROOT / 'books/_prototypes/vol1-web-animations/original-art-catalog/build.py')
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path
    books = root / 'books'
    extension = books / '_extensions/mlsysbook-ext/figure-explore'
    extension.mkdir(parents=True)
    monkeypatch.setattr(catalog, 'ROOT', root)
    monkeypatch.setattr(catalog, 'BOOKS', books)
    source = 'books/vol1/01_source/01_source.qmd'
    qmd = root / source
    qmd.parent.mkdir(parents=True)
    qmd.write_text('::: {#fig-test fig-cap="Exact source"}\nOriginal artwork\n:::\n')
    impl = {'name': 'original-test', 'script': 'original-test.js', 'stylesheet': 'original-test.css', 'shared_player': True}
    paths = [extension / name for name in ['original-test.js', 'original-test.css', 'original-player.js', 'original-player.css']]
    for path in paths:
        path.write_text('/* ' + path.name + ' */')
    art = root / 'original.svg'
    art.write_text('<svg/>')
    key = 'vol1:fig-test'
    entry = {'source': source, 'volume': 1, 'figure_id': 'fig-test',
             'runtime_sha1': sha1(b'\x00'.join(p.read_bytes() for p in paths)).hexdigest(),
             'source_sha1': sha1(qmd.read_bytes()).hexdigest(),
             'art_sha1': sha1(art.read_bytes()).hexdigest(), 'art_structural_sha1': 'structural-digest'}
    run = books / '_build/parallel/final'
    run.mkdir(parents=True)
    # The deliberately nonstandard job name proves paths use summary.job.
    job = {'job': 'exact-output-job', 'format': 'html', 'volume': 'vol1',
           'chapter': '01_source', 'ok': True, 'returncode': 0}
    output = run / job['job'] / 'output'
    chapter = output / Path(source).relative_to('books').with_suffix('.html')
    chapter.parent.mkdir(parents=True)
    chapter.write_text('<div id="fig-test" data-mlsys-explore="original-test" data-mlsys-explore-key="vol1:fig-test"></div><script src="../../site_libs/original-test.js?cache=1"></script>')
    emitted = output / 'site_libs/original-test.js'
    emitted.parent.mkdir()
    emitted.write_bytes(paths[0].read_bytes())
    record = {field: entry[field] for field in ('runtime_sha1', 'source_sha1', 'art_sha1', 'art_structural_sha1')}
    record.update(build_root='books/_build/parallel/final', human_student_testing=False)
    summary = run / 'summary.json'
    summary.write_text(json.dumps({'jobs': [job]}))
    return SimpleNamespace(root=root, books=books, extension=extension, source=source, qmd=qmd,
                           impl=impl, art=art, key=key, entry=entry, run=run, job=job,
                           chapter=chapter, emitted=emitted, record=record, summary=summary)


def current(p):
    return catalog.implementation_review_current(p.key, p.record, {p.key: p.entry}, p.impl,
                                                  {p.key: {'asset': str(p.art)}}, p.run)


def links(p):
    return catalog.verified_chapter_links(p.run, {p.key: p.entry}, {p.key: p.impl})


@pytest.mark.parametrize('chapter', ['01_source', None])
def test_successful_owning_chapter_and_full_volume_allow_review_and_exact_link(project, chapter):
    p = project
    p.job['chapter'] = chapter
    p.summary.write_text(json.dumps({'jobs': [p.job]}))
    assert current(p)
    assert catalog.rendered_source_chapter(p.run, p.entry, p.impl) == p.chapter
    assert links(p) == {p.key: '/_build/parallel/final/exact-output-job/output/vol1/01_source/01_source.html?view=explore#fig-test'}


@pytest.mark.parametrize('field,value', [('chapter', '02_other'), ('volume', 'vol2'), ('format', 'pdf'), ('ok', False), ('returncode', 1)])
def test_unrelated_or_failed_job_never_establishes_review_or_link(project, field, value):
    p = project
    p.job[field] = value
    p.summary.write_text(json.dumps({'jobs': [p.job]}))
    assert not current(p)
    assert links(p) == {}


@pytest.mark.parametrize('chapter', ['01_source', None])
def test_successful_summary_without_actual_source_html_is_insufficient(project, chapter):
    p = project
    p.job['chapter'] = chapter
    p.summary.write_text(json.dumps({'jobs': [p.job]}))
    p.chapter.unlink()
    # An unrelated HTML file must not substitute for the exact source route.
    (p.chapter.parent / 'another.html').write_text('Unrelated chapter')
    assert not current(p)
    assert links(p) == {}


@pytest.mark.parametrize('replacement', [
    '<div id="fig-test" data-mlsys-explore="old-player" data-mlsys-explore-key="vol1:fig-test"></div><script src="../../site_libs/original-test.js"></script>',
    '<div id="fig-test" data-mlsys-explore="original-test" data-mlsys-explore-key="vol1:fig-wrong"></div><script src="../../site_libs/original-test.js"></script>',
    '<div id="fig-other" data-mlsys-explore="original-test" data-mlsys-explore-key="vol1:fig-test"></div><script src="../../site_libs/original-test.js"></script>',
    '<div id="fig-test" data-mlsys-explore="original-test" data-mlsys-explore-key="vol1:fig-test"></div>',
    '<div id="fig-test" data-mlsys-explore="original-test" data-mlsys-explore-key="vol1:fig-test"></div><script src="../../site_libs/original-test.js.stale"></script>',
    '<div id="fig-test" data-mlsys-explore="original-test" data-mlsys-explore-key="vol1:fig-test"></div><script src="https://other.test/original-test.js"></script>',
])
def test_stale_or_absent_actual_marker_or_script_blocks_review_and_link(project, replacement):
    p = project
    p.job['chapter'] = None
    p.summary.write_text(json.dumps({'jobs': [p.job]}))
    p.chapter.write_text(replacement)
    assert not current(p)
    assert links(p) == {}


def test_missing_emitted_script_is_not_a_mounted_implementation(project):
    p = project
    p.emitted.unlink()
    assert not current(p)
    assert links(p) == {}


@pytest.mark.parametrize('artifact', ['qmd', 'art', 'module'])
def test_source_art_or_runtime_drift_expires_existing_review(project, artifact):
    p = project
    path = p.extension / 'original-test.js' if artifact == 'module' else getattr(p, artifact)
    path.write_bytes(path.read_bytes() + b'\nchanged')
    assert not current(p)


def test_explicit_other_build_never_reuses_existing_review(project):
    p = project
    other = p.books / '_build/parallel/other'
    other.mkdir()
    assert not catalog.implementation_review_current(p.key, p.record, {p.key: p.entry}, p.impl,
                                                      {p.key: {'asset': str(p.art)}}, other)


def test_only_successful_matching_candidate_can_supply_route(project):
    p = project
    failed = {**p.job, 'job': 'missing-output', 'returncode': 1, 'ok': False}
    whole = {**p.job, 'chapter': None}
    p.summary.write_text(json.dumps({'jobs': [failed, whole]}))
    assert current(p)
    assert p.key in links(p)


def test_job_cannot_escape_the_declared_build(project):
    p = project
    p.job['job'] = '../elsewhere'
    p.summary.write_text(json.dumps({'jobs': [p.job]}))
    assert not current(p)
    assert links(p) == {}


@pytest.fixture
def mixed_volumes(project):
    p = project
    followup = p.books / '_build/parallel/followup'
    shutil.copytree(p.run, followup)
    first = {**p.record, 'build_root': 'books/_build/parallel/followup'}
    source = 'books/vol2/01_source/01_source.qmd'
    qmd = p.root / source
    qmd.parent.mkdir(parents=True)
    qmd.write_bytes(p.qmd.read_bytes())
    second = {**p.entry, 'source': source, 'volume': 2}
    job = {**p.job, 'job': 'second-volume', 'volume': 'vol2', 'chapter': None}
    output = p.run / job['job'] / 'output'
    chapter = output / Path(source).relative_to('books').with_suffix('.html')
    chapter.parent.mkdir(parents=True)
    chapter.write_text(p.chapter.read_text().replace('vol1:fig-test', 'vol2:fig-test'))
    emitted = output / 'site_libs/original-test.js'
    emitted.parent.mkdir()
    emitted.write_bytes(p.emitted.read_bytes())
    p.summary.write_text(json.dumps({'jobs': [p.job, job]}))
    registry = {p.key: p.entry, 'vol2:fig-test': second}
    manifest = {key: p.impl for key in registry}
    records = {p.key: first, 'vol2:fig-test': p.record}
    art = {key: {'asset': str(p.art)} for key in registry}
    roots = {'vol1': 'books/_build/parallel/followup', 'vol2': 'books/_build/parallel/final'}
    return p, followup, registry, manifest, records, art, roots


def test_volume_map_uses_current_review_and_exact_route_for_each_run(mixed_volumes):
    p, followup, registry, manifest, records, art, roots = mixed_volumes
    reviewed, ready = catalog.catalog_review_state(registry, manifest, records, art, build_root_map=roots)
    assert reviewed == set(manifest)
    assert '/parallel/followup/' in ready[p.key]
    assert '/parallel/final/second-volume/' in ready['vol2:fig-test']
    wrong = {**roots, 'vol1': roots['vol2']}
    reviewed, _ = catalog.catalog_review_state(registry, manifest, records, art, build_root_map=wrong)
    assert reviewed == {'vol2:fig-test'}


def test_volume_map_does_not_fall_back_to_an_older_available_chapter(mixed_volumes):
    p, followup, registry, manifest, records, art, roots = mixed_volumes
    (followup / p.chapter.relative_to(p.run)).unlink()
    reviewed, ready = catalog.catalog_review_state(registry, manifest, records, art, build_root_map=roots)
    assert reviewed == {'vol2:fig-test'}
    assert set(ready) == {'vol2:fig-test'}


def test_volume_map_does_not_refresh_a_stale_source_fingerprint(mixed_volumes):
    p, _, registry, manifest, records, art, roots = mixed_volumes
    p.qmd.write_text(p.qmd.read_text() + '\nChanged source context')
    reviewed, _ = catalog.catalog_review_state(registry, manifest, records, art, build_root_map=roots)
    assert reviewed == {'vol2:fig-test'}


@pytest.mark.parametrize('roots', [{'vol1': 'run'}, {'vol1': 'run', 'vol2': ''}, {'vol1': 'run', 'vol2': 'run', 'vol3': 'run'}])
def test_volume_map_requires_an_explicit_complete_assignment(mixed_volumes, roots):
    _, _, registry, manifest, records, art, _ = mixed_volumes
    with pytest.raises(ValueError, match='exactly the enrolled volumes'):
        catalog.catalog_review_state(registry, manifest, records, art, build_root_map=roots)
