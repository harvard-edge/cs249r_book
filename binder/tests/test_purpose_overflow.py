"""The opener fit gate must reject overflow and incomplete evidence (2026-09-11)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from binder.cli.commands import _pdf_checks as checks
from binder.cli.commands.layout import LayoutCommand
from binder.cli.core.discovery import ChapterDiscovery

ROOT = Path(__file__).resolve().parents[2]
HOOK = 'Why must a physical machine finish its computation before it moves?'
PARAGRAPH = 'The controller observes the environment and issues a command within its available time budget.'
CALLOUT = 'An independent enforcement path checks every proposed command before the actuator can apply it.'


def fixture_proof(tmp_path, monkeypatch, pages, *, labeled=False, extra_callout=False):
    books = tmp_path / 'books'
    (books / 'config').mkdir(parents=True)
    first = books / 'vol4/01_boundary/01_boundary.qmd'
    first.parent.mkdir(parents=True)
    first.write_text(
        '# The Causal Boundary {#sec-boundary}\n\n'
        + ('## Purpose {.unnumbered .unlisted}\n\n' if labeled else '')
        + '![](cover.svg){fig-alt="Architecture diagram"}\n\n'
        + '\\begin{marginfigure}\n\\stack{1}{2}\n\\end{marginfigure}\n\n'
        + f'*{HOOK}*\n\n{PARAGRAPH}\n\n'
        + ('::: {.callout-definition}\n' + CALLOUT + '\n:::\n\n' if extra_callout else '')
        + '{{< pagebreak >}}\n\n::: {.callout-learning-objectives}\nExplain the mechanism.\n:::\n'
        + '## First section\nThis body text is beyond the opener budget.\n')
    second = books / 'vol4/02_body/02_body.qmd'
    second.parent.mkdir(parents=True)
    second.write_text('# The Physical Body\n\n*How does the body respond to each command?*\n\n'
                      'A mechanical response requires enough time and available energy to complete safely.\n\n{{< pagebreak >}}\n')
    config = {'book': {'chapters': ['index.qmd', {'part': 'Anatomy', 'chapters': [
        first.relative_to(books).as_posix(), second.relative_to(books).as_posix()]}],
        'appendices': ['vol4/backmatter/references.qmd']}}
    (books / 'config/_quarto-pdf-vol4.yml').write_text(yaml.safe_dump(config))
    pdf = tmp_path / 'proof.pdf'
    pdf.write_bytes(b'mocked PDF')
    monkeypatch.setattr(checks.shutil, 'which', lambda name: '/mock/pdftotext')
    monkeypatch.setattr(checks.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='\f'.join(pages)))
    monkeypatch.setattr(LayoutCommand, '_load_chapter_map', staticmethod(lambda path: ([(1, 'The Causal Boundary')], [])))
    return pdf, first, second


@pytest.mark.parametrize('labeled', (False, True))
def test_labeled_and_untitled_openers_fit(tmp_path, monkeypatch, labeled):
    pdf, _, _ = fixture_proof(tmp_path, monkeypatch, [HOOK + '\n' + PARAGRAPH], labeled=labeled)
    assert checks.scan_purpose_overflow(pdf, 'vol4', tmp_path, chapter='causal boundary') == []


def test_paragraph_spills_to_next_page(tmp_path, monkeypatch):
    pdf, _, _ = fixture_proof(tmp_path, monkeypatch, [HOOK, PARAGRAPH])
    issues = checks.scan_purpose_overflow(pdf, 'vol4', tmp_path, chapter='01_boundary')
    assert [i.code for i in issues] == ['purpose-overflow']
    assert 'sheet 2' in issues[0].message and 'sheet 1' in issues[0].message


def test_entire_hook_and_paragraph_moved_off_cover_page_is_overflow(tmp_path, monkeypatch):
    pdf, _, _ = fixture_proof(tmp_path, monkeypatch, ['The Causal Boundary\nCover image', HOOK + '\n' + PARAGRAPH])
    issues = checks.scan_purpose_overflow(pdf, 'vol4', tmp_path, chapter='01_boundary')
    assert len(issues) == 2 and all(i.code == 'purpose-overflow' for i in issues)


def test_trailing_opener_callout_also_must_fit(tmp_path, monkeypatch):
    pdf, _, _ = fixture_proof(tmp_path, monkeypatch, [HOOK + '\n' + PARAGRAPH, CALLOUT], extra_callout=True)
    issues = checks.scan_purpose_overflow(pdf, 'vol4', tmp_path, chapter='01_boundary')
    assert [i.code for i in issues] == ['purpose-overflow']
    assert 'paragraph' in issues[0].message


def test_full_volume_gate_rejects_missing_expected_chapter(tmp_path, monkeypatch):
    pdf, _, _ = fixture_proof(tmp_path, monkeypatch, [HOOK + '\n' + PARAGRAPH])
    issues = checks.scan_purpose_overflow(pdf, 'vol4', tmp_path)
    assert [i.code for i in issues] == ['purpose-missing-chapter']
    assert '02_body' in issues[0].message


def test_stale_or_unmatched_source_is_not_silently_accepted(tmp_path, monkeypatch):
    pdf, _, _ = fixture_proof(tmp_path, monkeypatch, [HOOK + '\nAn older paragraph with different wording.'])
    issues = checks.scan_purpose_overflow(pdf, 'vol4', tmp_path, chapter='01_boundary')
    assert [i.code for i in issues] == ['purpose-anchor-missing']


def test_missing_bookmarks_prevent_false_pass(tmp_path, monkeypatch):
    pdf, _, _ = fixture_proof(tmp_path, monkeypatch, [HOOK + '\n' + PARAGRAPH])
    monkeypatch.setattr(LayoutCommand, '_load_chapter_map', staticmethod(lambda path: ([], [])))
    issues = checks.scan_purpose_overflow(pdf, 'vol4', tmp_path, chapter='01_boundary')
    assert [i.code for i in issues] == ['purpose-unverified']


def test_unknown_opener_structure_prevents_false_pass(tmp_path, monkeypatch):
    pdf, first, _ = fixture_proof(tmp_path, monkeypatch, [HOOK + '\n' + PARAGRAPH])
    first.write_text('# The Causal Boundary\nNo identifiable hook or opener.\n')
    issues = checks.scan_purpose_overflow(pdf, 'vol4', tmp_path, chapter='01_boundary')
    assert [i.code for i in issues] == ['purpose-source-unrecognized']


def test_scope_must_be_canonical_and_in_selected_volume(tmp_path, monkeypatch):
    pdf, _, _ = fixture_proof(tmp_path, monkeypatch, [HOOK + '\n' + PARAGRAPH])
    issues = checks.scan_purpose_overflow(pdf, 'vol4', tmp_path, chapter='vol1/01_boundary')
    assert [i.code for i in issues] == ['purpose-unverified']
    assert 'outside vol4' in issues[0].message


def test_normalization_handles_index_marks_ligatures_and_line_hyphenation():
    source = r'An **efficient** real-time controller\index{Controller!definition} observes its environment.'
    rendered = 'An efﬁcient real-\ntime controller observes its environ-\nment.'
    assert checks._purpose_normalize(source) == checks._purpose_normalize(rendered)


@pytest.mark.parametrize('volume', ('vol1', 'vol2', 'vol3', 'vol4'))
def test_source_parser_recognizes_every_current_main_chapter(volume):
    paths = checks._purpose_chapter_paths(ROOT, volume)
    assert len(paths) >= 16
    assert all(checks._purpose_source_anchors(path.read_text()) for path in paths)


@pytest.mark.parametrize('volume', ('vol1', 'vol2', 'vol3', 'vol4'))
def test_layout_cli_accepts_all_volumes_and_chapter_scope(tmp_path, volume):
    command = LayoutCommand(SimpleNamespace(root_dir=tmp_path), ChapterDiscovery(tmp_path))
    command._purpose = Mock(return_value=True)
    assert command.run(['purpose', 'proof.pdf', '--' + volume, '--chapter', 'causal boundary'])
    command._purpose.assert_called_once_with(Path('proof.pdf'), volume, chapter='causal boundary')
