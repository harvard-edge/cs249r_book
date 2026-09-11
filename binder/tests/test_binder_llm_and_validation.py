import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from rich.console import Console

from binder.cli.commands._index_checks import check_tag_placement
from binder.cli.commands.validate import ValidateCommand
import binder.cli.main as m


def test_build_json_output_parseable_and_unwrapped():
    """Ensure `binder build --json` produces valid, parseable JSON without wrapping."""
    cli = object.__new__(m.MLSysBookCLI)
    cli.config_manager = SimpleNamespace(show_active_config=lambda: None)
    cli.build_command = SimpleNamespace(
        build_volume=lambda *a, **kw: True,
        _last_build_log="/a/" + "x" * 100 + ".log",
    )
    f = io.StringIO()
    orig_console = m.console
    try:
        m.console = Console(file=f, width=80, color_system=None)
        ok = cli.handle_build_command(["html", "--vol1", "--json"])
        assert ok is True

        raw = f.getvalue()
        # The output in console must be directly parseable as JSON without errors
        data = json.loads(raw)
        assert data["success"] is True
        assert data["volume"] == "vol1"
        assert data["log_path"] == "/a/" + "x" * 100 + ".log"
        assert "\n" not in data["log_path"]
    finally:
        m.console = orig_console


def test_resolve_pdf_volumes_preserves_multiple_flags():
    """Ensure `--vol1 --vol2` preserves both volumes rather than early-returning first."""
    cmd = object.__new__(ValidateCommand)
    vols = cmd._resolve_pdf_volumes(Path("books/vol1"), Path("books"), vol1=True, vol2=True)
    assert vols == ["vol1", "vol2"]

    vols_single = cmd._resolve_pdf_volumes(Path("books/vol1"), Path("books"), vol1=True)
    assert vols_single == ["vol1"]

    vols_path = cmd._resolve_pdf_volumes(Path("books/vol3"), Path("books"))
    assert vols_path == ["vol3"]


def test_index_scanner_handles_single_qmd_file():
    """Ensure index scanner works when given an individual QMD file."""
    with TemporaryDirectory() as d:
        p = Path(d) / "chapter.qmd"
        p.write_text("## Heading \\index{Example}\n", encoding="utf-8")
        dir_issues = check_tag_placement(p.parent)
        file_issues = check_tag_placement(p)
        assert len(dir_issues) == 1
        assert len(file_issues) == 1
        assert file_issues[0].code == "V4_heading"
        assert file_issues[0].file == "chapter.qmd"


def test_content_tree_skips_outside_volume_collection():
    """Ensure content-tree validation is a no-op when path is not a volume collection."""
    cmd = object.__new__(ValidateCommand)
    # Chapter directory
    r_dir = cmd._run_content_tree(Path("books/vol1/introduction"))
    assert len(r_dir.issues) == 0
    assert r_dir.files_checked == 0

    # Chapter file
    r_file = cmd._run_content_tree(Path("books/vol1/introduction/introduction.qmd"))
    assert len(r_file.issues) == 0
    assert r_file.files_checked == 0

    # Volume directory
    r_vol = cmd._run_content_tree(Path("books/vol1"))
    assert len(r_vol.issues) == 0
    assert r_vol.files_checked == 2


def test_build_json_redirects_nested_output_to_stderr():
    """Ensure real BuildCommand output is directed to stderr leaving stdout clean JSON."""
    from contextlib import redirect_stdout
    from unittest.mock import patch

    buf = io.StringIO()
    cli = m.MLSysBookCLI()
    with patch.object(cli.build_command, "_preflight_epub_hygiene", return_value=False):
        with redirect_stdout(buf):
            cli.handle_build_command(["epub", "--vol1", "--json"])

    data = json.loads(buf.getvalue())
    assert data["success"] is False
    assert data["format"] == "epub"
    assert data["volume"] == "vol1"


def test_check_xref_resolves_across_containing_volume():
    """Ensure xrefs resolve against containing volume when scanning a single file."""
    from binder.cli.commands._index_checks import check_xref_resolves
    p = Path("books/vol1/introduction/introduction.qmd")
    if p.is_file():
        issues = check_xref_resolves(p)
        assert len(issues) == 0


def test_maintain_volume_bib_rejects_unsupported_volumes():
    """Ensure binder fix bib rejects vol1/vol2 with explanatory message."""
    from binder.cli.commands.maintenance import MaintenanceCommand
    cmd = object.__new__(MaintenanceCommand)
    assert cmd._maintain_volume_bib("vol1") is False
    assert cmd._maintain_volume_bib("vol2") is False


def test_volume_bibliographies_preserves_missing_dedicated_bib():
    """Ensure dedicated bibliographies for vol3/vol4 are preserved even when missing."""
    from binder.cli.checks.case_study_provenance import collect
    with TemporaryDirectory() as tmp:
        p = Path(tmp)
        (p / "books/vol4").mkdir(parents=True)
        (p / "books/references.bib").write_text("@article{example2020,\n title={Example}\n}\n", encoding="utf-8")
        (p / "books/vol4/test.qmd").write_text("::: {.callout-case-study}\n@ example\nSource @example2020\nProvenance: example\n:::\n", encoding="utf-8")
        checked, findings = collect(repo=p)
        assert checked == 0
        assert len(findings) == 1
        assert findings[0].code == "missing-bibliography"
        assert findings[0].file.endswith("references-vol4.bib")


def test_build_json_emits_json_on_conflicting_volumes():
    """Ensure conflicting volume arguments return valid JSON error payload."""
    from contextlib import redirect_stdout
    buf = io.StringIO()
    cli = m.MLSysBookCLI()
    with redirect_stdout(buf):
        ok = cli.handle_build_command(["pdf", "--vol1", "--vol2", "--json"])
    assert ok is False
    data = json.loads(buf.getvalue())
    assert data["success"] is False
    assert "Select only one volume" in data.get("error", "")


def test_build_json_emits_json_on_chapters_with_all():
    """Ensure combining explicit chapters with --all returns valid JSON error payload."""
    from contextlib import redirect_stdout
    buf = io.StringIO()
    cli = m.MLSysBookCLI()
    with redirect_stdout(buf):
        ok = cli.handle_build_command(["html", "01_intro", "--all", "--json"])
    assert ok is False
    data = json.loads(buf.getvalue())
    assert data["success"] is False
    assert "Cannot combine explicit chapters with --all" in data.get("error", "")
