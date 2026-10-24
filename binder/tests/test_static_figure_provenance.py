"""Mutation tests for exact approved-prose and historical-art provenance guards."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture
def provenance(tmp_path, monkeypatch, request):
    """Copy only checked inputs; mutations never touch production source files."""
    repo = Path(__file__).resolve().parents[2]
    script = repo / 'binder/tools/scripts/check_static_figure_calcs.py'
    spec = importlib.util.spec_from_file_location('static_provenance_under_test', script)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    delivery = guard.STATIC_DELIVERY
    record_path = repo / guard.SECTION_REVIEW
    manifest = json.loads(delivery.read_text())
    record = json.loads(record_path.read_text())
    sources = {entry['source'] for entry in manifest['figures']}
    sources.update(entry['source'] for entry in record['files'])
    inputs = [repo / source for source in sources] + [record_path, delivery]
    inputs += [delivery.parent / entry['png'] for entry in manifest['figures']]
    integrated = getattr(request, "param", False)
    if integrated:
        inputs.append(repo / guard.INTEGRATION_REVIEW)
    for source in inputs:
        target = tmp_path / source.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if source.suffix == ".qmd":
            rev = "3dac1031c7" if integrated else "efd1999a73"
            target.write_bytes(subprocess.check_output(["git", "show", rev + ":" + str(source.relative_to(repo))], cwd=repo))
    original_git = guard.subprocess.check_output

    def reviewed_git_bytes(args, cwd):
        # The temporary fixture is not a repository. Read the real, immutable
        # git baseline while all current-source reads use the isolated tree.
        assert args[:2] == ['git', 'show']
        return original_git(args, cwd=repo)

    monkeypatch.setattr(guard.subprocess, 'check_output', reviewed_git_bytes)
    monkeypatch.setattr(guard, 'ROOT', tmp_path)
    monkeypatch.setattr(guard, 'STATIC_DELIVERY', tmp_path / delivery.relative_to(repo))
    if not integrated:
        guard.static_source_provenance(refresh=True)
    return guard, tmp_path / record_path.relative_to(repo), record


def test_historical_proofs_match_reviewed_source_and_pngs(provenance):
    guard, _, _ = provenance
    result = guard.static_source_provenance()
    assert result['checked_figures'] == 26
    assert result['historical_raw_hashes_match']
    assert result['normalized_static_sources_equal_after_approved_reversal']
    assert result['reviewed_png_hashes_match']


@pytest.mark.parametrize('mutation, message', [
    ('outside_prose', 'unapproved post-edit source drift'),
    ('stale_edit', 'stale or ambiguous approved replacement'),
    ('stale_post_hash', 'unapproved post-edit source drift'),
    ('stale_pre_hash', 'approved pre-edit hash mismatch'),
    ('figure_with_fresh_post_hash', 'approved pre-edit hash mismatch'),
    ('protected_structure', 'replacement touches protected source'),
    ('inline_math', 'inline math/code changed'),
])
def test_approved_record_cannot_hide_source_drift(provenance, mutation, message):
    guard, record_path, record = provenance
    entry = record['files'][0]
    source_path = guard.ROOT / entry['source']
    current = source_path.read_bytes()
    edit = entry['edits'][0]
    if mutation == 'outside_prose':
        source_path.write_bytes(current + b'\nUnapproved ordinary prose.\n')
    elif mutation == 'stale_edit':
        edit['after'] += ' stale'
    elif mutation == 'stale_post_hash':
        entry['after_sha256'] = '0' * 64
    elif mutation == 'stale_pre_hash':
        entry['before_sha256'] = '0' * 64
    elif mutation == 'figure_with_fresh_post_hash':
        changed = current.replace(b'fig-cap="', b'fig-cap="Unapproved caption: ', 1)
        assert changed != current
        source_path.write_bytes(changed)
        entry['after_sha256'] = hashlib.sha256(changed).hexdigest()
    elif mutation in {'protected_structure', 'inline_math'}:
        old = edit['after']
        suffix = ('\n\n::: {#fig-forbidden}\nProtected\n:::'
                  if mutation == 'protected_structure' else ' New value $x$.')
        edit['after'] += suffix
        changed = current.replace(old.encode(), edit['after'].encode())
        assert changed != current
        source_path.write_bytes(changed)
        entry['after_sha256'] = hashlib.sha256(changed).hexdigest()
    record_path.write_text(json.dumps(record))
    with pytest.raises(AssertionError, match=message):
        guard.static_source_provenance()


def test_review_status_updates_do_not_change_exact_edit_binding(provenance):
    guard, record_path, record = provenance
    before = guard.static_source_provenance()['approved_prose_edits_sha256']
    record['status'] = 'Updated review metadata only'
    record_path.write_text(json.dumps(record))
    assert guard.static_source_provenance()['approved_prose_edits_sha256'] == before


def test_reviewed_png_drift_is_rejected(provenance):
    guard, _, _ = provenance
    manifest = json.loads(guard.STATIC_DELIVERY.read_text())
    png = guard.STATIC_DELIVERY.parent / manifest['figures'][0]['png']
    png.write_bytes(png.read_bytes() + b'changed')
    with pytest.raises(AssertionError, match='reviewed PNG changed'):
        guard.static_source_provenance()


@pytest.mark.parametrize('provenance', [True], indirect=True)
def test_integrated_sources_preserve_reviewed_figures(provenance):
    guard, _, _ = provenance
    assert guard.static_source_provenance()['checked_figures'] == 26


@pytest.mark.parametrize('provenance', [True], indirect=True)
@pytest.mark.parametrize('tamper', ['prose', 'figure', 'baseline'])
def test_integration_binding_rejects_unrecorded_changes(provenance, tamper):
    guard, _, _ = provenance
    record_path = guard.ROOT / guard.INTEGRATION_REVIEW
    record = json.loads(record_path.read_text())
    entry = record['files'][0]
    path = guard.ROOT / entry['source']
    if tamper == 'baseline':
        entry['before_sha256'] = '0' * 64
        record_path.write_text(json.dumps(record))
        message = 'integration baseline hash mismatch'
    else:
        raw = path.read_bytes()
        changed = raw + b'\nUnrecorded prose.\n' if tamper == 'prose' else raw.replace(b'fig-cap="', b'fig-cap="Changed ', 1)
        assert changed != raw
        path.write_bytes(changed)
        if tamper == 'figure':
            entry['after_sha256'] = hashlib.sha256(changed).hexdigest()
            record_path.write_text(json.dumps(record))
        message = 'unapproved integrated source drift' if tamper == 'prose' else 'reviewed figure changed during integration'
    with pytest.raises(AssertionError, match=message):
        guard.pre_integration_source(entry['source'])
