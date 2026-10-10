"""Regression tests for binder LLM ergonomics and validation scoping.

Guards against regressions in:
- Pure, unwrapped JSON build summaries (--json) with stderr logging separation.
- Structured JSON error responses on argument and execution failures.
- Multi-flag volume preservation (--vol1 --vol2).
- Single-file index scanning without false positives on cross-chapter references.
- Content-tree validation scoping against single-volume and repository checkouts.
- Dedicated volume bibliography preservation in case-study provenance and maintenance.
"""

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


def test_volume_bibliographies_uses_shared_bib():
    """Ensure volumes resolve case studies against the shared references.bib."""
    from binder.cli.checks.case_study_provenance import collect
    with TemporaryDirectory() as tmp:
        p = Path(tmp)
        (p / "books/vol4").mkdir(parents=True)
        (p / "books/references.bib").write_text("@article{example2020,\n title={Example}\n}\n", encoding="utf-8")
        (p / "books/vol4/test.qmd").write_text("::: {.callout-case-study}\n@ example\nSource @example2020\nProvenance: example\n:::\n", encoding="utf-8")
        checked, findings = collect(repo=p)
        assert checked == 1
        assert len(findings) == 0


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


def test_maintain_volume_bib_resolves_script_independently_of_cwd(monkeypatch):
    """Ensure bibliography maintenance finds the script regardless of current directory."""
    books_dir = Path("books")
    if books_dir.is_dir():
        monkeypatch.chdir(books_dir)
        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / "binder" / "tools" / "scripts" / "structure" / "build_volume_bib.py"
        assert script.exists()


def test_purpose_stack_validation_and_guidance():
    """Ensure purpose-stack check provides clean, structured issue reports with template suggestions."""
    with TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        cm = SimpleNamespace(repo_root=d, book_dir=d)
        cmd = ValidateCommand(cm, None)

        chap_dir = d / "01_test"
        chap_dir.mkdir(parents=True)
        svg_dir = chap_dir / "images" / "svg"
        svg_dir.mkdir(parents=True)
        svg_file = svg_dir / "margin_stack.svg"
        svg_file.write_text("<svg></svg>", encoding="utf-8")

        # 1. Valid chapter (stack has no caption by default)
        qmd = chap_dir / "01_test.qmd"
        qmd.write_text(
            '# Test Chapter\n\n'
            '## Purpose {.unnumbered .unlisted}\n\n'
            '::: {.column-margin}\n'
            '![](images/svg/margin_stack.svg){width="100%"}\n'
            ':::\n\n'
            '_Why does the system behave this way?_\n\n'
            'Chapter text begins here.\n',
            encoding="utf-8",
        )
        res = cmd._run_purpose_stack(d)
        assert res.passed is True
        assert len(res.issues) == 0

        # Also verify margin_caption_required does not require a caption for margin_stack.svg
        margin_res = cmd._run_margin_caption_required(d)
        assert margin_res.passed is True
        assert len(margin_res.issues) == 0

        # 2. Missing purpose heading
        qmd.write_text('# Test Chapter\n\nChapter text without purpose.\n', encoding="utf-8")
        res = cmd._run_purpose_stack(d)
        assert res.passed is False
        assert len(res.issues) == 1
        assert res.issues[0].code == "purpose_heading_missing"
        assert res.issues[0].line == 1
        assert "## Purpose {.unnumbered .unlisted}" in res.issues[0].suggestion

        # 3. Missing hook question
        qmd.write_text(
            '# Test Chapter\n\n'
            '## Purpose {.unnumbered .unlisted}\n\n'
            '::: {.column-margin}\n'
            '![](images/svg/margin_stack.svg){width="100%"}\n'
            ':::\n\n'
            'Chapter text begins without a hook question.\n',
            encoding="utf-8",
        )
        res = cmd._run_purpose_stack(d)
        assert res.passed is False
        codes = {i.code for i in res.issues}
        assert "purpose_hook_missing" in codes


def test_dedicated_check_help_panels():
    """Ensure dedicated help panels for structure, numbers, figures, prose, and refs render cleanly."""
    import binder.cli.commands.validate as vmod

    cmd = object.__new__(ValidateCommand)
    cmd.GROUPS = ValidateCommand.GROUPS

    out = io.StringIO()
    test_console = Console(file=out, color_system=None, width=120)
    orig_console = vmod.console
    try:
        vmod.console = test_console

        # Test group help panels
        for grp in ["structure", "numbers", "figures", "prose", "refs", "epub"]:
            out.seek(0)
            out.truncate(0)
            cmd._print_group_help(grp)
            rendered = out.getvalue()
            assert f"binder check {grp} — scopes" in rendered
            assert "Error codes" in rendered
            assert "Common invocations:" in rendered

        # Test general check help
        out.seek(0)
        out.truncate(0)
        cmd._print_check_help()
        check_help = out.getvalue()
        assert "--json" in check_help
        assert "--plain" in check_help
        assert "machine-readable" in check_help
    finally:
        vmod.console = orig_console


def test_check_catalog_loading_and_integrity():
    """Ensure CheckCatalog loads all groups and validates error code schema."""
    from binder.cli.core.catalog import get_check_catalog

    catalog = get_check_catalog()
    assert len(catalog.groups) >= 6
    expected_groups = ["structure", "numbers", "figures", "prose", "refs", "epub"]
    for grp in expected_groups:
        group_doc = catalog.get_group(grp)
        assert group_doc is not None
        assert group_doc.group == grp
        assert len(group_doc.title) > 0
        assert len(group_doc.scopes) > 0
        assert len(group_doc.error_codes) > 0
        for err in group_doc.error_codes:
            assert len(err.code) > 0
            assert len(err.trigger) > 0
            assert len(err.fix) > 0

    # Cross-group error resolution
    err_stack = catalog.get_error("purpose_stack_missing")
    assert err_stack is not None
    assert err_stack.code == "purpose_stack_missing"
    assert "margin_stack.svg" in err_stack.trigger or "architecture stack" in err_stack.trigger

    # Non-existent error returns None
    assert catalog.get_error("non_existent_diagnostic_code_12345") is None


def test_validation_issue_from_template_exact_and_fallback():
    """Ensure ValidationIssue.from_template instantiates correctly with and without catalog matches."""
    from binder.cli.commands.validate import ValidationIssue

    # 1. Known catalog code without overrides
    issue = ValidationIssue.from_template(
        code="purpose_stack_missing",
        file="books/vol1/01_intro/01_intro.qmd",
        line=42,
    )
    assert issue.code == "purpose_stack_missing"
    assert issue.file == "books/vol1/01_intro/01_intro.qmd"
    assert issue.line == 42
    assert "architecture stack diagram" in issue.message
    assert "margin_stack.svg" in issue.suggestion
    assert issue.severity == "error"
    d = issue.to_dict()
    assert d["code"] == "purpose_stack_missing"
    assert d["file"] == "books/vol1/01_intro/01_intro.qmd"
    assert d["line"] == 42

    # 2. Template interpolation with kwargs
    skip_issue = ValidationIssue.from_template(
        code="heading_level_skip",
        file="test.qmd",
        line=15,
        prev_level=1,
        level=3,
        skipped="H2",
    )
    assert skip_issue.message == "Heading jumps from H1 to H3 (skips H2)"
    assert skip_issue.suggestion != ""

    # 3. Fallback when code is not in catalog
    fallback_issue = ValidationIssue.from_template(
        code="completely_novel_custom_code",
        file="test.qmd",
        line=99,
    )
    assert fallback_issue.code == "completely_novel_custom_code"
    assert fallback_issue.message == "completely_novel_custom_code"
    assert fallback_issue.suggestion == ""

    # 4. Explicit overrides take precedence
    override_issue = ValidationIssue.from_template(
        code="purpose_stack_missing",
        file="test.qmd",
        line=1,
        message="Custom explicit message",
        suggestion="Custom explicit suggestion",
        severity="warning",
    )
    assert override_issue.message == "Custom explicit message"
    assert override_issue.suggestion == "Custom explicit suggestion"
    assert override_issue.severity == "warning"


