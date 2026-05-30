"""Tests for PDF overfull-box and margin-overflow detection.

Covers:
  * Log-based vertical-overflow gate (Overfull \\vbox) — warning-level, so it
    surfaces in the build without flipping result.ok (book build stays green).
  * The pure margin-overflow geometry used by `binder layout margins`.

The log-based hbox gate and the cross-ref/traceback scans predate this change
and are exercised indirectly here only where they intersect severity handling.
"""

from book.cli.commands._pdf_checks import (
    PdfCheckItem,
    PdfIssue,
    PdfValidationResult,
    format_checklist,
    scan_build_log,
)
from book.cli.commands.layout import LayoutCommand


# --------------------------------------------------------------------------
# Log-based vbox gate (A1)
# --------------------------------------------------------------------------

def _write_log(tmp_path, text):
    p = tmp_path / "latex-build.log"
    p.write_text(text, encoding="utf-8")
    return p


def test_vbox_overflow_emitted_as_warning(tmp_path):
    log = _write_log(
        tmp_path,
        "Overfull \\vbox (48.5pt too high) has occurred while \\output is active [17]\n"
        "Overfull \\vbox (22.0pt too high) detected [33]\n"
        "Overfull \\vbox (5.0pt too high) has occurred while \\output is active [40]\n",
    )
    issues = scan_build_log(log)
    vbox = [i for i in issues if i.code == "overfull-vbox"]
    assert len(vbox) == 1
    assert vbox[0].severity == "warning"
    # 48.5 and 22.0 are >= 20pt; 5.0 is excluded.
    assert vbox[0].count == 2
    assert "17" in vbox[0].message and "33" in vbox[0].message
    assert "40" not in vbox[0].message  # sub-threshold page not reported


def test_hbox_stays_error_level(tmp_path):
    log = _write_log(
        tmp_path,
        "Overfull \\hbox (35.0pt too wide) in paragraph at lines 100--102\n",
    )
    issues = scan_build_log(log)
    hbox = [i for i in issues if i.code == "overfull-hbox"]
    assert len(hbox) == 1
    assert hbox[0].severity == "error"


def test_no_vbox_when_all_below_threshold(tmp_path):
    log = _write_log(
        tmp_path,
        "Overfull \\vbox (3.0pt too high) has occurred while \\output is active [9]\n",
    )
    assert not [i for i in scan_build_log(tmp_path / "latex-build.log")
               if i.code == "overfull-vbox"]


# --------------------------------------------------------------------------
# severity → result.ok and checklist rendering
# --------------------------------------------------------------------------

def _result(issues, checks):
    return PdfValidationResult(
        volume="vol2", pdf_path=tmp_pdf(), issues=issues, checks=checks
    )


def tmp_pdf():
    from pathlib import Path
    return Path("Machine-Learning-Systems-Vol2.pdf")


def test_warning_only_keeps_result_ok():
    res = _result(
        issues=[PdfIssue(code="overfull-vbox", message="m", severity="warning")],
        checks=[PdfCheckItem("overfull-vbox", "No vbox", passed=False, is_warning=True)],
    )
    assert res.ok is True


def test_error_issue_fails_result_ok():
    res = _result(
        issues=[PdfIssue(code="overfull-hbox", message="m", severity="error")],
        checks=[PdfCheckItem("overfull-hbox", "No hbox", passed=False)],
    )
    assert res.ok is False


def test_checklist_renders_warning_section():
    res = _result(
        issues=[PdfIssue(code="overfull-vbox", message="margin overflow", severity="warning")],
        checks=[PdfCheckItem("overfull-vbox", "No vbox", passed=False, is_warning=True)],
    )
    out = format_checklist(res)
    assert "Warnings (non-blocking):" in out
    assert "⚠" in out
    assert "Issues:" not in out  # no error-level issues


# --------------------------------------------------------------------------
# margin-overflow geometry (A2) — pure, no PDF needed
# --------------------------------------------------------------------------

PW, PH = 600.0, 800.0   # footer band top = 752; margin column starts at x=330


def _char(x0, bottom):
    return {"x0": x0, "bottom": bottom, "top": bottom - 8}


def _img(x0, bottom):
    return {"x0": x0, "bottom": bottom, "top": bottom - 40}


def test_margin_image_past_footer_is_flagged():
    over_c, over_i = LayoutCommand._page_overflow(
        PW, PH, chars=[], images=[_img(400, 775)], tol=2.0
    )
    assert len(over_i) == 1 and not over_c


def test_main_column_content_below_footer_is_not_margin_overflow():
    # x0=100 is the main column, not the margin — handled by `collisions`, not here.
    over_c, over_i = LayoutCommand._page_overflow(
        PW, PH, chars=[_char(100, 775)], images=[_img(100, 775)], tol=2.0
    )
    assert not over_c and not over_i


def test_margin_content_above_footer_is_clean():
    over_c, over_i = LayoutCommand._page_overflow(
        PW, PH, chars=[_char(400, 740)], images=[_img(400, 700)], tol=2.0
    )
    assert not over_c and not over_i


def test_tolerance_respected():
    # bottom=755 is 3pt below the 752 footer line.
    over_tight, _ = LayoutCommand._page_overflow(
        PW, PH, chars=[_char(400, 755)], images=[], tol=2.0
    )
    over_loose, _ = LayoutCommand._page_overflow(
        PW, PH, chars=[_char(400, 755)], images=[], tol=10.0
    )
    assert len(over_tight) == 1   # 755 > 752+2
    assert len(over_loose) == 0   # 755 < 752+10


def test_scan_margin_overflow_missing_file_returns_none():
    from book.cli.commands.layout import scan_margin_overflow
    assert scan_margin_overflow("/no/such/file.pdf") is None
