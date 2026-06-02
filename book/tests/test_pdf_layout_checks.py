"""Tests for PDF overfull-box and margin-overflow detection.

Covers:
  * Log-based vertical-overflow gate (Overfull \\vbox) — warning-level, so it
    surfaces in the build without flipping result.ok (book build stays green).
  * The pure margin-overflow geometry used by `binder layout margins`.

The log-based hbox gate and the cross-ref/traceback scans predate this change
and are exercised indirectly here only where they intersect severity handling.
"""

from pathlib import Path

from book.cli.commands._pdf_checks import (
    PdfCheckItem,
    PdfIssue,
    PdfValidationResult,
    format_checklist,
    scan_build_log,
)
from book.cli.commands.layout import LayoutCommand
from book.cli.commands.layout import CollisionFinding


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


def _char(x0, bottom, text="x", x1=None):
    return {
        "x0": x0,
        "x1": x0 + 6 if x1 is None else x1,
        "bottom": bottom,
        "top": bottom - 8,
        "text": text,
    }


def _img(x0, bottom):
    # Narrow margin figure (~1.25in = 90pt wide).
    return {"x0": x0, "x1": x0 + 90, "bottom": bottom, "top": bottom - 40}


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


def test_full_width_line_is_not_margin_overflow():
    # A code-listing / wide-table line: leftmost char in the main column,
    # rightmost char crosses past the 55% margin line. The right fragment must
    # NOT be flagged — it is main-column content, not a margin note. (This was
    # the dominant false positive: full-width listings dipping low.)
    chars = [_char(100, 775), _char(400, 775)]  # same baseline
    over_c, _ = LayoutCommand._page_overflow(PW, PH, chars=chars, images=[], tol=2.0)
    assert not over_c


def test_full_width_image_straddling_margin_is_not_flagged():
    # A figure spanning the text block (x0 left of the margin, wide) is main
    # content even if it dips low; only narrow margin figures count.
    wide = {"x0": 80, "x1": 520, "bottom": 775, "top": 600}
    _, over_i = LayoutCommand._page_overflow(PW, PH, chars=[], images=[wide], tol=2.0)
    assert not over_i


def test_narrow_margin_figure_past_footer_still_flagged():
    # A genuine margin figure: starts in the margin, ~1.25in (90pt) wide.
    fig = {"x0": 400, "x1": 490, "bottom": 775, "top": 660}
    _, over_i = LayoutCommand._page_overflow(PW, PH, chars=[], images=[fig], tol=2.0)
    assert len(over_i) == 1


def test_text_flung_below_page_edge_is_excluded():
    # Figure-internal label placed off-canvas (bottom well past page height) is
    # not a margin caption clipping at the edge — exclude from the text signal.
    over_c, _ = LayoutCommand._page_overflow(
        PW, PH, chars=[_char(400, PH + 120)], images=[], tol=2.0
    )
    assert not over_c


def test_caption_clipping_at_page_edge_is_flagged():
    # A margin caption dipping into the footer band but still on-page → flagged.
    over_c, _ = LayoutCommand._page_overflow(
        PW, PH, chars=[_char(400, PH - 5)], images=[], tol=2.0
    )
    assert len(over_c) == 1


def test_scan_margin_overflow_missing_file_returns_none():
    from book.cli.commands.layout import scan_margin_overflow
    assert scan_margin_overflow("/no/such/file.pdf") is None


def test_chapter_filter_matches_title_slug_and_substring():
    assert LayoutCommand._matches_chapter("Inference at Scale", ["Inference at Scale"])
    assert LayoutCommand._matches_chapter("Inference at Scale", ["inference-at-scale"])
    assert LayoutCommand._matches_chapter("Distributed Training Systems", ["training"])
    assert not LayoutCommand._matches_chapter("Inference at Scale", ["Model Training"])


def test_source_map_is_scoped_to_pdf_volume(tmp_path):
    root = tmp_path
    vol1 = root / "book" / "quarto" / "contents" / "vol1" / "introduction"
    vol2 = root / "book" / "quarto" / "contents" / "vol2" / "introduction"
    vol1.mkdir(parents=True)
    vol2.mkdir(parents=True)
    (vol1 / "introduction.qmd").write_text("# Introduction\n", encoding="utf-8")
    (vol2 / "introduction.qmd").write_text("# Introduction\n", encoding="utf-8")
    pdf_dir = root / "book" / "quarto" / "_build" / "pdf-vol1"
    pdf_dir.mkdir(parents=True)

    source_map = LayoutCommand(None, None)._build_source_map(
        pdf_dir / "Machine-Learning-Systems-Vol1.pdf"
    )

    assert source_map["Introduction"] == (
        Path("book/quarto/contents/vol1/introduction/introduction.qmd")
    )


def test_margin_baseline_crowding_uses_baseline_gap_not_bbox_overlap():
    crowded = [
        _char(400, 108, text="first line", x1=455),
        _char(400, 110, text="second line", x1=460),
    ]
    normal = [
        _char(400, 108, text="first line", x1=455),
        _char(400, 116, text="second line", x1=460),
    ]

    assert len(LayoutCommand._margin_baseline_crowding(PW, crowded)) == 1
    assert not LayoutCommand._margin_baseline_crowding(PW, normal)


def test_margin_image_text_overlap_ignores_tiny_icons_and_flags_big_images():
    chars = [_char(400, 108, text="substantial label", x1=480)]
    tiny_icon = {"x0": 400, "x1": 410, "top": 96, "bottom": 112}
    big_image = {"x0": 390, "x1": 500, "top": 96, "bottom": 130}

    assert not LayoutCommand._margin_image_text_overlaps(PW, chars, [tiny_icon])
    assert len(LayoutCommand._margin_image_text_overlaps(PW, chars, [big_image])) == 1


def test_collision_csv_is_machine_readable(capsys):
    finding = CollisionFinding(
        sheet=74,
        label="66",
        chapter="Inference at Scale",
        band="footer",
        y=688.4,
        snippet="Bursts starve shared KV cache.",
    )

    LayoutCommand(None, None)._render_collisions_csv([finding])

    out = capsys.readouterr().out
    assert "chapter,sheet,label,band,y,snippet" in out
    assert "Inference at Scale,74,66,footer,688.4" in out
