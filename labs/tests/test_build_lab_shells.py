"""Production shells use canonical curriculum metadata and safe transient sources."""
import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / 'tools/build_lab_shells.py'
spec = importlib.util.spec_from_file_location('build_lab_shells', TOOL)
shells = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shells)
LABS = TOOL.parents[1]


def test_every_notebook_has_one_canonical_route():
    entries = shells.discover(LABS)
    assert {entry['volume'] for entry in entries} == {1, 2, 3, 4}
    assert len(entries) == len(list(LABS.glob('vol*/lab_*.py')))
    assert all(shells.chapter_link(LABS, entry) or entry['chapter'] == 0 for entry in entries)


def test_route_opens_a_fullpage_notebook():
    rows = [row for row in shells.discover(LABS) if row['volume'] == 3]
    page = shells.shell(LABS, rows[1], rows[0], rows[2])
    assert 'sidebar: false' in page
    assert 'window.location.replace("app/index.html"' in page
    assert 'navbar-common' not in page
    assert 'quarto-header' not in page
    assert 'https://mlsysbook.ai/vol3/introduction/introduction.html' in page
    assert '../lab_00_orientation/' in page
    assert '../lab_02_foundation_model/' in page
    assert '<iframe' not in page
    head, nav = shells.fullpage_navigation(LABS, rows[1], rows[0], rows[2])
    assert 'Lab course navigation' in nav
    assert '../../../index.html' in nav
    assert '../../lab_02_foundation_model/' in nav


def test_cleanup_unlinks_only_its_generated_files(tmp_path):
    manifest = tmp_path / 'manifest.json'
    generated = tmp_path / 'vol1/lab_test/index.qmd'
    generated.parent.mkdir(parents=True)
    generated.write_text(shells.MARKER)
    sibling = generated.parent / 'keep.md'
    sibling.write_text('authored')
    manifest.write_text('{"files":["vol1/lab_test/index.qmd"]}')
    shells.cleanup(tmp_path, manifest)
    assert not generated.exists()
    assert sibling.read_text() == 'authored'
    generated.write_text('authored replacement')
    with pytest.raises(ValueError, match='replaced'):
        shells.cleanup(tmp_path, manifest)
    assert generated.exists()


def test_cleanup_rejects_escape_from_labs_root(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{"files":["../index.qmd"]}')
    with pytest.raises(ValueError, match='Unsafe'):
        shells.cleanup(tmp_path, manifest)


def test_landing_inventory_is_generated_from_every_catalog(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOL.parent))
    import build_lab_inventory
    entries = shells.discover(LABS)
    content = build_lab_inventory.inventory(LABS)
    assert content == (LABS / 'config/lab-inventory.qmd').read_text()
    assert content.count('<details>') == len({row['volume'] for row in entries})
    assert content.count('<li><a href=') == len(entries)
    for row in entries:
        assert f'href="{row["route"]}/"' in content


def test_fullpage_layout_reserves_navigation_height():
    row = shells.discover(LABS)[0]
    head, nav = shells.fullpage_navigation(LABS, row, None, None)
    assert '100dvh - var(--lab-course-height)' in head
    assert 'assets/scripts/notebook-chrome.js' in nav
    assert 'assets/scripts/theme-persist.js' in head
    assert '<iframe' not in head + nav


def test_install_produces_direct_app_and_idempotent_navigation(tmp_path, monkeypatch):
    row = shells.discover(LABS)[1]
    monkeypatch.setattr(shells, 'discover', lambda labs: [row])
    target = tmp_path / row['route'] / 'app/index.html'
    target.parent.mkdir(parents=True)
    target.write_text('<!doctype html><html><head></head><body><div id="root"></div></body></html>')
    monkeypatch.setattr(shells, 'shared_header', lambda output: ('<style>/* shared */</style>', '<header id="quarto-header"><nav>Shared book menu</nav></header>'))
    shells.install(LABS, tmp_path)
    once = target.read_text()
    shells.install(LABS, tmp_path)
    assert target.read_text() == once
    assert once.count('id="lab-course-nav"') == 1
    assert '<iframe' not in once
    launcher = (target.parent.parent / 'index.html').read_text()
    assert 'window.location.replace' in launcher
    assert '<iframe' not in launcher


def test_shared_header_reuses_generated_navigation_and_scopes_styles(tmp_path):
    css = tmp_path / 'site_libs/bootstrap/bootstrap-test.css'
    css.parent.mkdir(parents=True)
    css.write_text(':root{--brand:red}body{margin:0}.navbar{color:var(--brand)}')
    (tmp_path / 'index.html').write_text('<link href="site_libs/bootstrap/bootstrap-test.css"><header id="quarto-header"><nav class="navbar"><a href="./index.html">Machine Learning Systems</a><a href="https://mlsysbook.ai/vol1/">Read</a></nav></header>')
    head, nav = shells.shared_header(tmp_path)
    assert 'Machine Learning Systems' in nav and 'https://mlsysbook.ai/vol1/' in nav
    assert 'href="../../../index.html"' in nav
    scoped = css.with_name('lab-chrome-light.css').read_text()
    assert '@scope (#quarto-header)' in scoped
    assert ':root' not in scoped and 'body{' not in scoped
    assert 'data-lab-chrome="light"' in head


def test_every_volume_starts_with_onboarding(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOL.parent))
    import build_lab_inventory
    entries = shells.discover(LABS)
    content = build_lab_inventory.inventory(LABS)
    for volume in {row['volume'] for row in entries}:
        first = next(row for row in entries if row['volume'] == volume)
        assert first['chapter'] == 0
        assert f'aria-label="Start Volume {shells.ROMAN[volume]} onboarding at Lab 00"' in content
    assert content.count('>Start Lab 00 ') == 4


def test_inventory_rejects_missing_onboarding(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOL.parent))
    import build_lab_inventory
    entries = [row for row in shells.discover(LABS) if not (row['volume'] == 2 and row['chapter'] == 0)]
    monkeypatch.setattr(build_lab_inventory, 'discover', lambda labs: entries)
    with pytest.raises(ValueError, match='Volume II must start with Lab 00'):
        build_lab_inventory.inventory(LABS)
