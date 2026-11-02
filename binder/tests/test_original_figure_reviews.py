"""Role reviews never outlive their source, art, module, runtime, or evidence."""
from argparse import Namespace
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('original_reviews', ROOT / 'binder/tools/scripts/review_original_figure.py')
reviews = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reviews)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


@pytest.fixture
def project(tmp_path):
    root = tmp_path
    entry = root / reviews.PROTOTYPES / 'original-v1-fixture/entry.json'
    impl = {'name': 'original-fixture', 'script': 'module.js', 'stylesheet': 'module.css',
            'shared_player': True, 'bundle': 'data.js', 'data_script': 'data.js',
            'data_scripts': ['extra.js', 'data.js'], 'scope': 'Toy input', 'question': 'What moves?'}
    dump(entry, {'key': 'vol1:fig-fixture', 'implementation': impl, 'purpose': 'Track identity',
                 'storyboard': ['Copy', 'Read'], 'guard': 'No measured timing'})
    source = 'books/vol1/01_fixture/01_fixture.qmd'
    qmd = root / source
    qmd.parent.mkdir(parents=True)
    qmd.write_text('Text before\n::: {#fig-fixture fig-cap="Original source"}\nOriginal art\n:::\n')
    dump(root / reviews.CATALOG, {'figures': [{'key': 'vol1:fig-fixture', 'source': source,
                                              'figure_id': 'fig-fixture', 'volume': 1}]})
    extension = root / reviews.EXTENSION
    extension.mkdir(parents=True)
    for name in ['module.js', 'module.css', 'data.js', 'extra.js', 'original-player.js',
                 'original-player.css', 'figure-explore.js', 'figure-views.js', 'figure-views.css']:
        (extension / name).write_text(f'/* {name} */')
    timing = root / 'books/config/shared/html/figure-explore.yml'
    timing.parent.mkdir(parents=True)
    timing.write_text('movement-ms: 1800\n')
    art = root / 'art.svg'
    art.write_text('<svg xmlns="http://www.w3.org/2000/svg"><text>Original</text></svg>')
    art_index = root / 'art-index.json'
    dump(art_index, {'vol1:fig-fixture': {'source': source, 'asset': str(art)}})
    evidence = root / 'proof.txt'
    evidence.write_text('Inspected copy and ownership states in actual fixture.')
    return Namespace(root=root, entry=entry, qmd=qmd, extension=extension, timing=timing,
                     art=art, art_index=art_index, evidence=evidence)


def record(p, role='technical', scope='fixture', verdict='passed'):
    return reviews.record_review(Namespace(entry=p.entry, role=role, scope=scope,
                                          verdict=verdict, basis='Bounded independent review',
                                          evidence=[p.evidence], art_index=p.art_index), p.root)


def collect(p, scope='fixture'):
    return reviews.collect_readiness([p.entry], p.art_index, scope, p.root)


def reasons(result, role):
    return result['entries'][0]['roles'][role]['reasons']


def test_one_role_writes_only_own_record_and_two_current_roles_are_required(project):
    p = project
    before = {str(f): f.read_bytes() for f in p.root.rglob('*') if f.is_file()}
    assert collect(p)['accepted_keys'] == []
    result = record(p)
    assert result['reviewed_at_utc'].endswith('Z')
    assert collect(p)['accepted_keys'] == []
    assert reasons(collect(p), 'learning') == [{'code': 'missing-review', 'role': 'learning'}]
    assert not (p.entry.parent / 'learning-review.json').exists()
    assert set(str(f) for f in p.root.rglob('*') if f.is_file()) - set(before) == {str(p.entry.parent / 'technical-review.json')}
    assert all(Path(f).read_bytes() == data for f, data in before.items())
    record(p, 'learning')
    assert collect(p)['accepted_keys'] == ['vol1:fig-fixture']


def test_collection_is_read_only_and_default_discovery_works(project):
    p = project
    record(p); record(p, 'learning')
    before = {str(f): (f.read_bytes(), f.stat().st_mtime_ns) for f in p.root.rglob('*') if f.is_file()}
    assert reviews.collect_readiness(None, p.art_index, 'fixture', p.root)['accepted_keys'] == ['vol1:fig-fixture']
    assert all((Path(f).read_bytes(), Path(f).stat().st_mtime_ns) == value for f, value in before.items())


def test_production_never_inferred_from_fixture_and_blocked_role_blocks(project):
    p = project
    record(p); record(p, 'learning')
    assert collect(p, 'production')['accepted_keys'] == []
    assert reasons(collect(p, 'production'), 'technical')[0]['code'] == 'insufficient-scope'
    record(p, scope='production'); record(p, 'learning', 'production', 'blocked')
    assert collect(p, 'production')['accepted_keys'] == []
    assert reasons(collect(p, 'production'), 'learning')[0]['code'] == 'blocked-verdict'
    record(p, 'learning', 'production')
    assert collect(p, 'production')['accepted_keys'] == ['vol1:fig-fixture']
    assert collect(p, 'fixture')['accepted_keys'] == ['vol1:fig-fixture']


@pytest.mark.parametrize('artifact,component', [
    ('qmd', 'source'), ('art', 'canonical_art'), ('entry', 'entry_semantics_sha256'),
    ('module.js', 'module_assets'), ('module.css', 'module_assets'),
    ('data.js', 'module_assets'), ('extra.js', 'module_assets'),
    ('original-player.js', 'runtime_assets'), ('original-player.css', 'runtime_assets'),
    ('figure-explore.js', 'runtime_assets'), ('figure-views.js', 'runtime_assets'),
    ('figure-views.css', 'runtime_assets'), ('timing', 'timing'),
])
def test_every_bound_artifact_change_invalidates_both_roles(project, artifact, component):
    p = project
    record(p); record(p, 'learning')
    path = getattr(p, artifact) if artifact in ('qmd', 'art', 'entry', 'timing') else p.extension / artifact
    if artifact == 'entry':
        data = json.loads(path.read_text()); data['purpose'] = 'New teaching purpose'; dump(path, data)
    else:
        path.write_bytes(path.read_bytes() + b'\nchanged')
    result = collect(p)
    assert result['accepted_keys'] == []
    for role in reviews.ROLES:
        assert reasons(result, role) == [{'code': 'stale-artifacts', 'changed': [component]}]
    # Shared-only revalidation never silently carries the opposing role forward.
    record(p)
    assert collect(p)['accepted_keys'] == []
    assert not reasons(collect(p), 'technical')


def test_source_normalization_ignores_only_enrollment_and_not_caption_or_prose(project):
    p = project
    record(p); record(p, 'learning')
    p.qmd.write_text(p.qmd.read_text().replace('#fig-fixture ', '#fig-fixture data-explore="new-player" data-explore-key="vol1:fig-fixture" '))
    assert collect(p)['accepted_keys'] == ['vol1:fig-fixture']
    p.qmd.write_text(p.qmd.read_text().replace('Original source', 'Changed source'))
    assert collect(p)['accepted_keys'] == []
    data = b'Prose data-explore="keep"\n::: {#fig-x data-explore=\'remove\' fig-cap="data-explore=keep"}\n'
    assert reviews.normalized_source(data) == b'Prose data-explore="keep"\n::: {#fig-x fig-cap="data-explore=keep"}\n'


def test_semantic_entry_hash_ignores_json_formatting_and_aliases_bind_once(project):
    p = project
    current = reviews.current_artifacts(p.entry, p.art_index, p.root)
    assert len(current['artifacts']['module_assets']) == 4
    p.entry.write_text(json.dumps(json.loads(p.entry.read_text()), sort_keys=True))
    assert reviews.current_artifacts(p.entry, p.art_index, p.root)['artifact_fingerprint'] == current['artifact_fingerprint']


@pytest.mark.parametrize('change,code', [('changed', 'stale-evidence'), ('missing', 'missing-evidence')])
def test_evidence_changed_or_missing_invalidates_reviews(project, change, code):
    p = project
    record(p); record(p, 'learning')
    if change == 'changed': p.evidence.write_text('Different report')
    else: p.evidence.unlink()
    result = collect(p)
    assert result['accepted_keys'] == []
    assert reasons(result, 'technical')[0]['code'] == code


def test_invalid_source_provenance_and_missing_runtime_never_accepted(project):
    p = project
    data = json.loads(p.art_index.read_text()); data['vol1:fig-fixture']['source'] = 'wrong.qmd'; dump(p.art_index, data)
    with pytest.raises(reviews.ReviewError, match='source does not match'):
        record(p)
    result = collect(p)
    assert result['entries'][0]['reasons'][0]['code'] == 'invalid-current-artifacts'


def test_record_rejects_directories_as_evidence_and_does_not_touch_other_role(project):
    p = project
    record(p, 'learning')
    other = (p.entry.parent / 'learning-review.json').read_bytes()
    p.evidence = p.root
    with pytest.raises(reviews.ReviewError, match='individual file'):
        record(p)
    assert (p.entry.parent / 'learning-review.json').read_bytes() == other
    assert not (p.entry.parent / 'technical-review.json').exists()


def test_tampered_fingerprint_and_duplicate_entries_fail_closed(project):
    p = project
    record(p); record(p, 'learning')
    path = p.entry.parent / 'technical-review.json'
    data = json.loads(path.read_text()); data['artifact_fingerprint'] = '0' * 64; dump(path, data)
    assert reasons(collect(p), 'technical')[0]['code'] == 'invalid-review'
    record(p)
    result = reviews.collect_readiness([p.entry, p.entry], p.art_index, 'fixture', p.root)
    assert result['accepted_keys'] == []
    assert all(r['reasons'] == [{'code': 'duplicate-entry-key'}] for r in result['entries'])


def test_asset_path_escape_is_rejected_before_recording(project):
    p = project
    data = json.loads(p.entry.read_text()); data['implementation']['script'] = '../../outside.js'; dump(p.entry, data)
    with pytest.raises(reviews.ReviewError, match='asset filename'):
        record(p)
    assert not (p.entry.parent / 'technical-review.json').exists()


def test_cli_record_and_collect_emit_json_without_mutating_other_files(project, monkeypatch, capsys):
    p = project
    # main() resolves ROOT through function defaults, so run the CLI functions
    # through their same public argument parsing with a fixture-bound root.
    original_record, original_collect = reviews.record_review, reviews.collect_readiness
    monkeypatch.setattr(reviews, 'record_review', lambda args: original_record(args, p.root))
    monkeypatch.setattr(reviews, 'collect_readiness', lambda entries, art_index, scope: original_collect(entries, art_index, scope, p.root))
    for role in reviews.ROLES:
        monkeypatch.setattr('sys.argv', ['review_original_figure.py', 'record', '--entry', str(p.entry), '--role', role, '--verdict', 'passed', '--scope', 'production', '--basis', 'Actual chapter endpoints reviewed', '--evidence', str(p.evidence), '--art-index', str(p.art_index)])
        reviews.main()
        assert json.loads(capsys.readouterr().out)['role'] == role
    monkeypatch.setattr('sys.argv', ['review_original_figure.py', 'collect', '--entry', str(p.entry), '--art-index', str(p.art_index)])
    reviews.main()
    result = json.loads(capsys.readouterr().out)
    assert result['accepted_keys'] == ['vol1:fig-fixture']
    assert result['required_scope'] == 'production'


def test_missing_current_runtime_and_symlink_review_fail_closed(project):
    p = project
    record(p); record(p, 'learning')
    role = p.entry.parent / 'technical-review.json'
    copy = p.root / 'review-copy.json'
    copy.write_bytes(role.read_bytes()); role.unlink(); role.symlink_to(copy)
    assert reasons(collect(p), 'technical')[0]['code'] == 'invalid-review'
    with pytest.raises(reviews.ReviewError, match='symlink review destination'):
        record(p)
    (p.extension / 'original-player.js').unlink()
    result = collect(p)
    assert result['accepted_keys'] == []
    assert result['entries'][0]['reasons'][0]['code'] == 'invalid-current-artifacts'
