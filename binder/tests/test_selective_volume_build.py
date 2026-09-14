"""Regression coverage for volume-scoped chapter iteration (2026-09-11)."""

import signal
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from binder.cli.commands.build import BuildCommand
from binder.cli.core.config import ConfigManager, active_config_source
from binder.cli.core.discovery import AmbiguousChapterError, ChapterDiscovery
from binder.cli.core.volume_index import volume_index_source
from binder.cli.main import MLSysBookCLI

ROOT = Path(__file__).resolve().parents[2]
VOLUMES = ('vol1', 'vol2', 'vol3', 'vol4')


def fixture_book(tmp_path, volume='vol4', format_type='pdf'):
    books = tmp_path / 'books'
    (books / 'config').mkdir(parents=True)
    source = volume_index_source(books, volume, format_type)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b'# Preface\r\nCanonical introduction.\r\n')
    chapter = books / volume / '01_boundary/01_boundary.qmd'
    chapter.parent.mkdir(parents=True)
    chapter.write_text('# The Causal Boundary {#sec-boundary}\n\nPurpose.\n')
    unused = books / volume / '02_body/02_body.qmd'
    unused.parent.mkdir(parents=True)
    unused.write_text('# The Physical Body\n')
    config = books / 'config' / f'_quarto-{format_type}-{volume}.yml'
    config.write_text(yaml.safe_dump({
        'project': {'output-dir': f'_build/{format_type}-{volume}', 'render': ['index.qmd',
                    str(chapter.relative_to(books)), str(unused.relative_to(books))],
                    'post-render': ['shared/hooks.py']},
        'book': {'output-file': 'Whole-Volume', 'chapters': ['index.qmd', {'part': 'Anatomy',
                 'chapters': [str(chapter.relative_to(books)), str(unused.relative_to(books))]}],
                 'appendices': [f'{volume}/backmatter/references.qmd']},
        'format': {'titlepage-pdf': {'coverpage': True}},
        'metadata-files': ['config/shared.yml'],
    }, sort_keys=False))
    manager = ConfigManager(tmp_path)
    command = BuildCommand(manager, ChapterDiscovery(books))
    return books, source, chapter, config, command


@pytest.mark.parametrize('volume', VOLUMES)
@pytest.mark.parametrize('format_type', ('html', 'pdf', 'epub'))
def test_selected_render_uses_only_index_and_target_and_restores(tmp_path, volume, format_type):
    books, source, chapter, canonical, command = fixture_book(tmp_path, volume, format_type)
    original_config = canonical.read_bytes()
    original_source = source.read_bytes()
    active = books / '_quarto.yml'
    index = books / 'index.qmd'
    active.write_bytes(b'# Prior config\r\n')
    index.write_bytes(b'# Prior index\r\n')
    full_output = books / f'_build/{format_type}-{volume}/Whole-Volume.{format_type}'
    full_output.parent.mkdir(parents=True)
    full_output.write_bytes(b'KEEP FULL ARTIFACT')
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def fake_render(cmd, *, cwd, description):
        current = yaml.safe_load(active.read_text())
        assert cwd == books
        assert cmd == ['quarto', 'render', '--to=' + ('titlepage-pdf' if format_type == 'pdf' else format_type)]
        assert canonical.read_bytes() == original_config
        assert index.read_bytes() == original_source
        assert active_config_source(active) == canonical.relative_to(books).as_posix()
        expected = ['index.qmd', chapter.relative_to(books).as_posix()]
        assert current['project']['post-render'] == ['shared/hooks.py']
        if format_type == 'html':
            assert current['project']['render'] == expected
        else:
            assert current['book']['chapters'] == expected
            assert 'appendices' not in current['book']
            assert 'render' not in current['project']
            assert current['book']['output-file'] == '01_boundary'
        output = books / current['project']['output-dir']
        assert output == full_output.parent / 'chapters/01_boundary'
        (output / ('index.html' if format_type == 'html' else f'01_boundary.{format_type}')).write_bytes(b'BUILT')
        return True

    command._run_command = Mock(side_effect=fake_render)
    assert command.build_chapters_with_volume(['causal boundary'], format_type, volume,
                                             skip_hygiene=True, skip_validate=True)
    assert command._run_command.call_count == 1
    assert active.read_bytes() == b'# Prior config\r\n'
    assert index.read_bytes() == b'# Prior index\r\n'
    assert canonical.read_bytes() == original_config
    assert source.read_bytes() == original_source
    assert full_output.read_bytes() == b'KEEP FULL ARTIFACT'
    assert all(signal.getsignal(sig) == handler for sig, handler in handlers.items())
    assert not list((books / 'config').glob('*.backup'))


@pytest.mark.parametrize('outcome', ('failure', 'exception', 'interrupt', 'no-artifact'))
def test_failed_build_restores_legacy_symlink_and_absent_index(tmp_path, outcome):
    books, source, chapter, canonical, command = fixture_book(tmp_path)
    original = canonical.read_bytes()
    active = books / '_quarto.yml'
    active.symlink_to(canonical)

    def fail(*args, **kwargs):
        if outcome == 'exception':
            raise RuntimeError('Renderer crashed')
        if outcome == 'interrupt':
            raise KeyboardInterrupt()
        return outcome == 'no-artifact'

    command._run_command = fail
    assert not command.build_chapters_with_volume(['01_boundary'], 'pdf', 'vol4', skip_validate=True)
    assert active.is_symlink() and active.resolve() == canonical
    assert canonical.read_bytes() == original
    assert not (books / 'index.qmd').exists()


def test_preface_is_not_duplicated_when_it_is_generated_index(tmp_path):
    books, source, chapter, canonical, command = fixture_book(tmp_path)

    def render(*args, **kwargs):
        current = yaml.safe_load((books / '_quarto.yml').read_text())
        assert current['book']['chapters'] == ['index.qmd']
        output = books / current['project']['output-dir']
        (output / 'about.pdf').write_bytes(b'BUILT')
        return True

    command._run_command = render
    assert command.build_chapters_with_volume(['Preface'], 'pdf', 'vol4', skip_validate=True)


@pytest.mark.parametrize('query', ('vol1/01_boundary', 'vol1/01_boundary/01_boundary.qmd', 'missing chapter'))
def test_bad_scope_or_missing_chapter_never_renders(tmp_path, query):
    books, source, chapter, canonical, command = fixture_book(tmp_path)
    command._run_command = Mock()
    assert not command.build_chapters_with_volume([query], 'pdf', 'vol4')
    command._run_command.assert_not_called()
    assert not (books / '_quarto.yml').exists()


def test_missing_volume_config_cannot_fall_back(tmp_path):
    books, source, chapter, canonical, command = fixture_book(tmp_path)
    canonical.rename(books / 'config/_quarto-pdf-vol1.yml')
    command._run_command = Mock()
    assert not command.build_chapters_with_volume(['01_boundary'], 'pdf', 'vol4')
    command._run_command.assert_not_called()


@pytest.mark.parametrize('query', ('vol4/01_boundary', 'vol4/01_boundary.qmd',
                                  'vol4/01_boundary/01_boundary.qmd', 'vol4/the causal boundary',
                                  'vol4/causal-boundary', 'vol4/Causal Boundry'))
def test_filename_title_substring_and_typo_queries(tmp_path, query):
    books, source, chapter, canonical, command = fixture_book(tmp_path)
    assert command.chapter_discovery.find_chapter_file(query, allow_fuzzy=True) == chapter


def test_ambiguous_within_volume_is_not_chosen_by_shorter_name(tmp_path):
    books, source, chapter, canonical, command = fixture_book(tmp_path)
    other = chapter.with_name('03_boundary_safety.qmd')
    other.write_text('# The Boundary and Safety\n')
    with pytest.raises(AmbiguousChapterError) as caught:
        command.chapter_discovery.find_chapter_file('vol4/boundary', allow_fuzzy=True)
    assert len(caught.value.locations) == 2
    assert all(path.startswith('vol4/') for path in caught.value.locations)
    assert command.chapter_discovery.find_chapter_file('vol4/01_boundary') == chapter


def test_same_stem_across_volumes_requires_scope(tmp_path):
    books, source, chapter, canonical, command = fixture_book(tmp_path)
    other = books / 'vol3/01_boundary/01_boundary.qmd'
    other.parent.mkdir(parents=True)
    other.write_text('# Agentic Boundary\n')
    with pytest.raises(AmbiguousChapterError):
        command.chapter_discovery.find_chapter_file('01_boundary')
    assert command.chapter_discovery.find_chapter_file('vol3/01_boundary') == other


@pytest.mark.parametrize('volume', VOLUMES)
def test_parser_preserves_readable_query_and_volume(volume):
    cli = object.__new__(MLSysBookCLI)
    parsed = cli._parse_build_args(['pdf', '--' + volume, 'the', 'causal', 'boundary'])
    assert parsed[:4] == ('pdf', volume, False, 'the causal boundary')
    parsed = cli._parse_build_args(['pdf', 'intro', '--' + volume])
    assert parsed[:4] == ('pdf', volume, False, 'intro')


def test_parser_rejects_conflicting_volumes():
    cli = object.__new__(MLSysBookCLI)
    with pytest.raises(ValueError, match='only one volume'):
        cli._parse_build_args(['pdf', '--vol4', '--vol1'])


def test_prefixed_build_routes_to_volume_scoped_path(tmp_path):
    books, source, chapter, canonical, command = fixture_book(tmp_path)
    command.build_chapters_with_volume = Mock(return_value=True)
    assert command.build_chapters(['vol4/causal boundary'], 'pdf', skip_validate=True)
    command.build_chapters_with_volume.assert_called_once_with(
        ['vol4/01_boundary/01_boundary.qmd'], 'pdf', 'vol4', skip_hygiene=False, skip_validate=True)


def test_cli_prefixed_html_uses_volume_scoped_dispatch(tmp_path):
    books, source, chapter, canonical, command = fixture_book(tmp_path, format_type='html')
    command.build_chapters_with_volume = Mock(return_value=True)
    command.build_html_only = Mock(return_value=True)
    cli = object.__new__(MLSysBookCLI)
    cli.config_manager = Mock()
    cli.build_command = command
    assert cli.handle_build_command(['html', 'vol4/01_boundary/01_boundary.qmd'])
    command.build_chapters_with_volume.assert_called_once_with(
        ['vol4/01_boundary/01_boundary.qmd'], 'html', 'vol4', skip_hygiene=False, skip_validate=False)


@pytest.mark.parametrize('volume', VOLUMES)
def test_real_volume_discovery_resolves_all_canonical_stems(volume):
    discovery = ChapterDiscovery(ROOT / 'books')
    stems = discovery.get_chapters_from_config(volume)
    assert len(stems) > 10
    assert all(discovery.find_chapter_file(f'{volume}/{stem}') for stem in stems)


@pytest.mark.parametrize('verbose', (False, True))
def test_interruption_stops_renderer_group_before_restoration(tmp_path, monkeypatch, verbose):
    import os
    from binder.cli.commands import build as build_module

    books, source, chapter, canonical, command = fixture_book(tmp_path)
    command.verbose = verbose
    renderer = Mock(pid=987654321)
    renderer.poll.return_value = None
    renderer.communicate.side_effect = KeyboardInterrupt()
    renderer.stdout.readline.side_effect = KeyboardInterrupt()
    popen = Mock(return_value=renderer)
    monkeypatch.setattr(build_module.subprocess, 'Popen', popen)
    kill_group = Mock()
    if os.name == 'posix':
        monkeypatch.setattr(build_module.os, 'killpg', kill_group)
    with pytest.raises(KeyboardInterrupt):
        command._run_command(['quarto', 'render'], books, 'Test interrupt')
    assert popen.call_args.kwargs['start_new_session'] is True
    if os.name == 'posix':
        kill_group.assert_called_once_with(renderer.pid, signal.SIGKILL)
    else:
        renderer.kill.assert_called_once()
    renderer.wait.assert_called_once()


@pytest.mark.parametrize('build_kind', ('full', 'volume', 'selective'))
@pytest.mark.parametrize('verbose', (False, True))
@pytest.mark.parametrize('signum', (signal.SIGINT, signal.SIGTERM))
def test_build_signal_stops_renderer_before_shared_file_restoration(
        tmp_path, monkeypatch, build_kind, verbose, signum):
    """An interrupted build kills the renderer group, then restores shared files and handlers."""
    import os
    from binder.cli.commands import build as build_module

    books, source, chapter, canonical, command = fixture_book(tmp_path)
    command.verbose = verbose
    original = canonical.read_bytes()
    active = books / '_quarto.yml'
    index = books / 'index.qmd'
    active.write_bytes(b'# Prior config\n')
    index.write_bytes(b'# Prior index\n')
    header = books / 'shared/tex/header-includes.tex'
    header.parent.mkdir(parents=True)
    header.write_text('\\CropMarksfalse\n')
    events = []
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    monkeypatch.setattr(command.config_manager, 'get_config_file', lambda *args: canonical)

    renderer = Mock(pid=987654321)
    renderer.poll.return_value = None

    def interrupt(*args, **kwargs):
        signal.getsignal(signum)(signum, None)

    def wait_for_renderer(*args, **kwargs):
        events.append('wait')
        assert active.read_bytes() != b'# Prior config\n'
        if build_kind == 'volume':
            assert header.read_text() == '\\CropMarkstrue\n'

    renderer.communicate.side_effect = interrupt
    renderer.stdout.readline.side_effect = interrupt
    renderer.wait.side_effect = wait_for_renderer
    monkeypatch.setattr(build_module.subprocess, 'Popen', Mock(return_value=renderer))
    if os.name == 'posix':
        monkeypatch.setattr(build_module.os, 'killpg', lambda *args: events.append('kill'))
    else:
        renderer.kill.side_effect = lambda: events.append('kill')

    try:
        if build_kind == 'selective':
            assert not command.build_chapters_with_volume(['01_boundary'], 'pdf', 'vol4')
            assert active.read_bytes() == b'# Prior config\n'
            assert index.read_bytes() == b'# Prior index\n'
        elif build_kind == 'full':
            assert not command.build_full('pdf')
        else:
            assert not command.build_volume('vol4', 'pdf', print_marks=True)
        assert events == ['kill', 'wait']
        assert all(signal.getsignal(sig) == handler for sig, handler in previous_handlers.items())
        assert canonical.read_bytes() == original
        assert header.read_text() == '\\CropMarksfalse\n'
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def test_no_cover_edits_the_generated_config_not_the_source(tmp_path):
    """--no-cover used to edit the source YAML after it was copied, so it never took effect (2026-09-12)."""
    books, source, chapter, canonical, command = fixture_book(tmp_path)
    original = canonical.read_bytes()
    active = books / '_quarto.yml'

    def render(cmd, *, cwd, description):
        assert '    coverpage: false' in active.read_text()
        assert canonical.read_bytes() == original
        return True

    command._run_command = Mock(side_effect=render)
    assert command.build_volume('vol4', 'pdf', skip_validate=True, no_cover=True)
    assert command._run_command.call_count == 1
    assert canonical.read_bytes() == original


def test_print_marks_toggles_the_shared_header_and_restores_it(tmp_path):
    books, source, chapter, canonical, command = fixture_book(tmp_path)
    header = books / 'shared/tex/header-includes.tex'
    header.parent.mkdir(parents=True)
    header.write_text('\\newif\\ifCropMarks\n\\CropMarksfalse\n')

    def render(cmd, *, cwd, description):
        assert '\\CropMarkstrue' in header.read_text()
        return True

    command._run_command = Mock(side_effect=render)
    assert command.build_volume('vol4', 'pdf', skip_validate=True, print_marks=True)
    assert header.read_text() == '\\newif\\ifCropMarks\n\\CropMarksfalse\n'


def test_presentation_flags_are_rejected_for_non_pdf_volume_builds(tmp_path):
    books, source, chapter, canonical, command = fixture_book(tmp_path, format_type='html')
    command._run_command = Mock()
    assert not command.build_volume('vol4', 'html', no_cover=True)
    command._run_command.assert_not_called()


def test_chapters_from_several_volumes_are_rejected(tmp_path):
    books, source, chapter, canonical, command = fixture_book(tmp_path)
    other = books / 'vol3/05_memory/05_memory.qmd'
    other.parent.mkdir(parents=True)
    other.write_text('# Memory Hierarchy\n')
    command._run_command = Mock()
    command.build_chapters_with_volume = Mock(return_value=True)
    assert not command.build_chapters(['vol4/01_boundary', 'vol3/05_memory'], 'pdf')
    command.build_chapters_with_volume.assert_not_called()
    command._run_command.assert_not_called()
