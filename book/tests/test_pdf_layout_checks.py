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
    scan_pdf_text,
)
from book.cli.commands.layout import LayoutCommand
from book.cli.commands.layout import PageReport
from book.cli.commands.layout import CollisionFinding
from book.cli.checks.margin_geometry import MarginGeometryFinding
from book.cli.checks.margin_geometry import scan_page as scan_margin_geometry_page


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


def test_pdf_text_scan_flags_bare_crossrefs(monkeypatch, tmp_path):
    pdf = tmp_path / "artifact.pdf"
    pdf.write_bytes(b"%PDF placeholder")
    monkeypatch.setattr(
        "book.cli.commands._pdf_checks._pdftotext",
        lambda _: (
            "Adversarial training is covered in @sec-robust-ai. "
            "Decorators like @staticmethod, emails like a@b.com, and "
            "explanatory @citekey text are not Quarto cross-refs."
        ),
    )

    issues = scan_pdf_text(pdf)

    literal = [i for i in issues if i.code == "literal-crossref"]
    assert len(literal) == 1
    assert "@sec-robust-ai" in literal[0].message
    assert not [i for i in issues if i.code == "unresolved-crossref"]


def test_pdf_text_scan_flags_question_mark_crossrefs_once(monkeypatch, tmp_path):
    pdf = tmp_path / "artifact.pdf"
    pdf.write_bytes(b"%PDF placeholder")
    monkeypatch.setattr(
        "book.cli.commands._pdf_checks._pdftotext",
        lambda _: "Broken output shows ?@tbl-defense-selection-framework.",
    )

    issues = scan_pdf_text(pdf)

    assert len([i for i in issues if i.code == "unresolved-crossref"]) == 1
    assert not [i for i in issues if i.code == "literal-crossref"]


# --------------------------------------------------------------------------
# margin-overflow geometry (A2) — pure, no PDF needed
# --------------------------------------------------------------------------

PW, PH = 600.0, 800.0   # footer band top = 752


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
        PW, PH, chars=[], images=[_img(500, 775)], tol=2.0
    )
    assert len(over_i) == 1 and not over_c


def test_left_margin_content_past_footer_is_flagged():
    over_c, over_i = LayoutCommand._page_overflow(
        PW, PH, chars=[_char(40, 775)], images=[_img(30, 775)], tol=2.0
    )
    assert len(over_c) == 1 and len(over_i) == 1


def test_main_column_content_below_footer_is_not_margin_overflow():
    # x0=200 is the main column, not the margin — handled by `collisions`, not here.
    over_c, over_i = LayoutCommand._page_overflow(
        PW, PH, chars=[_char(200, 775)], images=[_img(200, 775)], tol=2.0
    )
    assert not over_c and not over_i


def test_margin_content_above_footer_is_clean():
    over_c, over_i = LayoutCommand._page_overflow(
        PW, PH, chars=[_char(500, 740)], images=[_img(500, 700)], tol=2.0
    )
    assert not over_c and not over_i


def test_tolerance_respected():
    # bottom=755 is 3pt below the 752 footer line.
    over_tight, _ = LayoutCommand._page_overflow(
        PW, PH, chars=[_char(500, 755)], images=[], tol=2.0
    )
    over_loose, _ = LayoutCommand._page_overflow(
        PW, PH, chars=[_char(500, 755)], images=[], tol=10.0
    )
    assert len(over_tight) == 1   # 755 > 752+2
    assert len(over_loose) == 0   # 755 < 752+10


def test_full_width_line_is_not_margin_overflow():
    # A code-listing / wide-table line in the main/body column should not be
    # flagged as margin content.
    chars = [_char(200, 775), _char(400, 775)]  # same baseline
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
    fig = {"x0": 500, "x1": 590, "bottom": 775, "top": 660}
    _, over_i = LayoutCommand._page_overflow(PW, PH, chars=[], images=[fig], tol=2.0)
    assert len(over_i) == 1


def test_text_flung_below_page_edge_is_excluded():
    # Figure-internal label placed off-canvas (bottom well past page height) is
    # not a margin caption clipping at the edge — exclude from the text signal.
    over_c, _ = LayoutCommand._page_overflow(
        PW, PH, chars=[_char(500, PH + 120)], images=[], tol=2.0
    )
    assert not over_c


def test_caption_clipping_at_page_edge_is_flagged():
    # A margin caption dipping into the footer band but still on-page → flagged.
    over_c, _ = LayoutCommand._page_overflow(
        PW, PH, chars=[_char(500, PH - 5)], images=[], tol=2.0
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


def test_source_line_match_normalizes_pdf_hyphenation(tmp_path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        "\n".join([
            "# Introduction",
            "",
            "[^fn-wsc]: **Warehouse-Scale Computer (WSC)**: Barroso and "
            "Holzle describe the datacenter as one system.",
        ]),
        encoding="utf-8",
    )

    assert LayoutCommand._find_source_line(
        qmd,
        "Ware-house-Scale Com-puter (WSC)",
    ) == 3


def test_source_line_match_strips_rendered_sidenote_number(tmp_path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        "\n".join([
            "# ML Systems",
            "",
            "HIPAA (Health Insurance Portability and Accountability Act) appears "
            "in prose before the rendered sidenote.",
            "",
            "[^fn-hipaa]: **HIPAA (Health Insurance Portability and "
            "Accountability Act)**: This US law translates security measures.",
        ]),
        encoding="utf-8",
    )

    assert LayoutCommand._find_source_line(
        qmd,
        "15 HIPAA (Health In- surance Portability and Accountability Act)",
    ) == 5


def test_source_line_match_prefers_footnote_title_over_nearby_mentions(tmp_path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        "\n".join([
            "# ML Systems",
            "",
            "[^fn-inference]: **Inference**: Mentions 2 W edge accelerators.",
            "",
            "[^fn-cloud-elastic-cost]: **Pay-as-You-Go Pricing**: Sustained "
            "workloads drive the total cost of ownership (TCO) analysis later.",
            "",
            "[^fn-tco]: [offset=65mm] **Total Cost of Ownership (TCO)**: "
            "Quantifies the gap between sticker price and true system cost.",
        ]),
        encoding="utf-8",
    )

    assert LayoutCommand._find_source_line(
        qmd,
        "16 Total Cost of Ownership (TCO) : Quantifies the gap",
    ) == 7


def test_source_line_match_ignores_short_bold_title_subset(tmp_path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        "\n".join([
            "# Neural Computation",
            "",
            "[^fn-inference]: **Inference**: This boundary changes which "
            "hardware is viable, from data center GPUs to 2 W edge accelerators.",
            "",
            "[^fn-edge]: [offset=-65mm] **Edge Inference Accelerators**: "
            "The Edge TPU operates in the mobile tier.",
        ]),
        encoding="utf-8",
    )

    assert LayoutCommand._find_source_line(
        qmd,
        "44 Edge Inference Ac- celerators : The Edge TPU",
    ) == 5


def test_source_line_match_allows_short_rendered_labels(tmp_path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        "\\node[right=12pt] {Execution\\\\ Opt.};\n",
        encoding="utf-8",
    )

    assert LayoutCommand._find_source_line(qmd, "Execution Opt.") == 1


def test_callout_break_suggestion_moves_existing_late_tcbbreak(tmp_path):
    qmd_dir = tmp_path / "book" / "quarto" / "contents" / "vol1" / "hw"
    qmd_dir.mkdir(parents=True)
    qmd = qmd_dir / "hw.qmd"
    qmd.write_text(
        "\n".join([
            ":::: {#nbk-throughput .callout-notebook title=\"The throughput ceiling\"}",
            "",
            "**The hardware constraints**",
            "",
            "- Peak compute",
            "",
            "**The workload characteristics**",
            "",
            "- Model",
            "- Data movement",
            "",
            "```{=latex}",
            "\\tcbbreak",
            "```",
            "",
            "**The prediction**",
            "",
            "::::",
        ]),
        encoding="utf-8",
    )

    cmd = LayoutCommand(_Cfg(tmp_path), None)
    suggestion = cmd._suggest_callout_break_fix(
        "book/quarto/contents/vol1/hw/hw.qmd",
        7,
        "The workload characteristics",
    )

    assert "move existing \\tcbbreak from line 13 to before line 7" in suggestion
    assert "The throughput ceiling" in suggestion


def test_callout_break_suggestion_inserts_at_semantic_label(tmp_path):
    qmd_dir = tmp_path / "book" / "quarto" / "contents" / "vol1" / "hw"
    qmd_dir.mkdir(parents=True)
    qmd = qmd_dir / "hw.qmd"
    qmd.write_text(
        "\n".join([
            ":::: {#chk-example .callout-checkpoint title=\"Checkpoint\"}",
            "",
            "**First section**",
            "",
            "- Item A",
            "",
            "**Second section**",
            "",
            "- Item B",
            "",
            "::::",
        ]),
        encoding="utf-8",
    )

    cmd = LayoutCommand(_Cfg(tmp_path), None)
    suggestion = cmd._suggest_callout_break_fix(
        "book/quarto/contents/vol1/hw/hw.qmd",
        8,
        "Item B",
    )

    assert "insert \\tcbbreak before line 7" in suggestion
    assert "callout-checkpoint" in suggestion


def test_callout_break_suggestion_moves_whole_callout_near_start(tmp_path):
    qmd_dir = tmp_path / "book" / "quarto" / "contents" / "vol1" / "hw"
    qmd_dir.mkdir(parents=True)
    qmd = qmd_dir / "hw.qmd"
    qmd.write_text(
        "\n".join([
            ":::: {#chk-example .callout-checkpoint title=\"Checkpoint\"}",
            "",
            "**First section**",
            "",
            "- Item A",
            "",
            "::::",
        ]),
        encoding="utf-8",
    )

    cmd = LayoutCommand(_Cfg(tmp_path), None)
    suggestion = cmd._suggest_callout_break_fix(
        "book/quarto/contents/vol1/hw/hw.qmd",
        3,
        "First section",
    )

    assert "move the whole callout or its lead-in earlier" in suggestion


def test_layout_strategy_routes_callout_source_context(tmp_path):
    qmd_dir = tmp_path / "book" / "quarto" / "contents" / "vol1" / "hw"
    qmd_dir.mkdir(parents=True)
    qmd = qmd_dir / "hw.qmd"
    qmd.write_text(
        "\n".join([
            ":::: {#chk-example .callout-checkpoint title=\"Checkpoint\"}",
            "",
            "**First section**",
            "",
            "- Item A",
            "",
            "**Second section**",
            "",
            "- Item B",
            "",
            "::::",
            "",
            "Some adjacent source-flow block.",
        ]),
        encoding="utf-8",
    )

    cmd = LayoutCommand(_Cfg(tmp_path), None)
    report = PageReport(
        sheet=1,
        label="1",
        chapter="Hardware",
        whitespace_pts=160,
        whitespace_frac=0.28,
        body_bottom_y=500,
        usable_bottom_y=660,
        cause="callout/box",
        detail="Second section",
        source_file="book/quarto/contents/vol1/hw/hw.qmd",
        source_line=7,
    )

    assert cmd._layout_strategy(report) == "callout-tcbbreak"

    report.source_line = 13
    assert cmd._layout_strategy(report) == "source-flow-callout-adjacent"

    report.source_file = ""
    report.source_line = 0
    assert cmd._layout_strategy(report) == "callout-localize"


def test_margin_geometry_strategy_routes_source_context(tmp_path):
    qmd_dir = tmp_path / "book" / "quarto" / "contents" / "vol1" / "hw"
    qmd_dir.mkdir(parents=True)
    qmd = qmd_dir / "hw.qmd"
    qmd.write_text(
        "\n".join([
            "# Hardware",
            "",
            "::: {.column-margin}",
            "![](images/svg/accel.svg){width=\"100%\"}",
            "*Accelerator margin caption.*",
            ":::",
            "",
            "[^fn-chip]: **Chip placement**: The note sits in the margin.",
        ]),
        encoding="utf-8",
    )

    cmd = LayoutCommand(_Cfg(tmp_path), None)
    finding = MarginGeometryFinding(
        page=4,
        issue="overflow-bottom",
        side="right",
        detail="text extends to 724pt, 28pt below text box",
        bbox=(420, 680, 520, 724),
        snippet="Chip placement",
    )
    base_row = {
        "finding": finding,
        "source_file": "book/quarto/contents/vol1/hw/hw.qmd",
        "source_line": 0,
    }

    row = dict(base_row, source_line=8)
    assert cmd._margin_geometry_strategy(row) == "margin-offset"

    row = dict(base_row, source_line=4)
    assert cmd._margin_geometry_strategy(row) == "margin-vspace"

    row = dict(base_row, source_file="", source_line=0)
    assert cmd._margin_geometry_strategy(row) == "margin-geometry-review"


def test_layout_plan_defers_margin_until_main_flow_is_clean(tmp_path):
    cmd = LayoutCommand(_Cfg(tmp_path), None)
    main = PageReport(
        sheet=10,
        label="8",
        chapter="Hardware",
        whitespace_pts=160,
        whitespace_frac=0.28,
        body_bottom_y=500,
        usable_bottom_y=660,
        cause="callout/box",
        detail="Second section",
        source_file="book/quarto/contents/vol1/hw/hw.qmd",
        source_line=7,
        layout_strategy="callout-tcbbreak",
    )
    margin_finding = MarginGeometryFinding(
        page=11,
        issue="overflow-bottom",
        side="right",
        detail="text extends to 724pt, 28pt below text box",
        bbox=(420, 680, 520, 724),
        snippet="Chip placement",
    )
    margin_row = {
        "finding": margin_finding,
        "label": "9",
        "chapter": "Hardware",
        "source_file": "book/quarto/contents/vol1/hw/hw.qmd",
        "source_line": 8,
        "section": "Accelerators",
    }

    plan = cmd._layout_plan_payload(
        volume="vol1",
        pdf_path=tmp_path / "vol1.pdf",
        whitespace_rows=[main],
        margin_rows=[margin_row],
        pages_scanned=20,
        page_count=20,
    )

    assert plan["workflow"]["next_phase"] == "1-main-flow"
    assert plan["counts"]["ready"] == 1
    assert plan["counts"]["deferred"] == 1
    main_item, margin_item = plan["items"]
    assert main_item["phase"] == "1-main-flow"
    assert main_item["ready"] is True
    assert margin_item["phase"] == "2-margin-calibration"
    assert margin_item["deferred"] is True
    assert margin_item["ready"] is False

    margin_only = cmd._layout_plan_payload(
        volume="vol1",
        pdf_path=tmp_path / "vol1.pdf",
        whitespace_rows=[],
        margin_rows=[margin_row],
        pages_scanned=20,
        page_count=20,
    )

    assert margin_only["workflow"]["next_phase"] == "2-margin-calibration"
    assert margin_only["items"][0]["deferred"] is False


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
        _char(500, 108, text="first line", x1=555),
        _char(500, 110, text="second line", x1=560),
    ]
    normal = [
        _char(500, 108, text="first line", x1=555),
        _char(500, 116, text="second line", x1=560),
    ]

    assert len(LayoutCommand._margin_baseline_crowding(PW, crowded)) == 1
    assert not LayoutCommand._margin_baseline_crowding(PW, normal)


def test_margin_image_text_overlap_ignores_tiny_icons_and_flags_big_images():
    chars = [_char(500, 108, text="substantial label", x1=560)]
    tiny_icon = {"x0": 500, "x1": 510, "top": 96, "bottom": 112}
    big_image = {"x0": 490, "x1": 590, "top": 96, "bottom": 130}

    assert not LayoutCommand._margin_image_text_overlaps(PW, chars, [tiny_icon])
    assert len(LayoutCommand._margin_image_text_overlaps(PW, chars, [big_image])) == 1


def test_left_margin_vector_text_overlap_is_flagged():
    chars = [_char(40, 108, text="substantial label", x1=110)]
    vector_box = {"x0": 30, "x1": 120, "top": 96, "bottom": 130}

    assert len(LayoutCommand._margin_image_text_overlaps(PW, chars, [vector_box])) == 1


# --------------------------------------------------------------------------
# Binder-native margin geometry (PyMuPDF-shaped fake page, no PDF needed)
# --------------------------------------------------------------------------

class _Rect:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.width = x1 - x0
        self.height = y1 - y0
        self.is_empty = self.width <= 0 or self.height <= 0


class _FakePage:
    def __init__(self, blocks=None, drawings=None):
        self._blocks = blocks or []
        self._drawings = [{"rect": _Rect(*d)} for d in (drawings or [])]

    def get_text(self, kind):
        assert kind == "dict"
        return {"blocks": self._blocks}

    def get_drawings(self):
        return self._drawings


def _text_block(x0, y0, x1, y1, text="margin text"):
    return {
        "type": 0,
        "bbox": (x0, y0, x1, y1),
        "lines": [{"spans": [{"text": text}]}],
    }


def test_native_margin_geometry_detects_real_graphic_text_overlap():
    page = _FakePage(
        blocks=[_text_block(470, 100, 555, 150, "overlapping margin note")],
        drawings=[(480, 130, 540, 180)],
    )

    findings = scan_margin_geometry_page(page, 12)

    assert [f.issue for f in findings] == ["overlap"]
    assert findings[0].page == 12
    assert "overlapping margin note" in findings[0].snippet


def test_native_margin_geometry_merges_adjacent_caption_with_figure():
    page = _FakePage(
        blocks=[_text_block(470, 144, 555, 164, "caption directly below")],
        drawings=[(480, 100, 540, 140)],
    )

    assert scan_margin_geometry_page(page, 12) == []


def test_native_margin_geometry_ignores_body_figure_internals_in_margin_band():
    page = _FakePage(
        blocks=[_text_block(70, 700, 105, 716, "axis label")],
        drawings=[
            (63, 680, 450, 760),  # full body/table graphic crossing the page edge
            (74, 704, 82, 714),   # internal icon in the physical margin band
        ],
    )

    assert scan_margin_geometry_page(page, 12) == []


def test_native_margin_geometry_flags_true_margin_bottom_overflow():
    page = _FakePage(
        blocks=[_text_block(470, 660, 555, 692, "caption below text box")],
    )

    findings = scan_margin_geometry_page(page, 12)

    assert [f.issue for f in findings] == ["overflow-bottom"]
    assert findings[0].side == "right"


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


# --------------------------------------------------------------------------
# table-only layout audit extraction — pure, no Quarto/PDF render needed
# --------------------------------------------------------------------------

class _Cfg:
    def __init__(self, root):
        self.root_dir = root
        self.book_dir = root / "book" / "quarto"


def test_pipe_table_extraction_preserves_label_colwidths_and_source(tmp_path):
    qmd_dir = (
        tmp_path
        / "book"
        / "quarto"
        / "contents"
        / "vol2"
        / "inference"
    )
    qmd_dir.mkdir(parents=True)
    qmd = qmd_dir / "inference.qmd"
    qmd.write_text(
        "\n".join([
            "# Inference",
            "",
            "| **Phase** | **Duration** |",
            "|:----------|-------------:|",
            "| Startup   | 5s           |",
            "| Warmup    | 15s          |",
            "",
            ": **Cold-Start Timeline**: Per-phase duration. "
            "{#tbl-cold-start tbl-colwidths=\"[60,40]\"}",
            "",
        ]),
        encoding="utf-8",
    )

    entries = LayoutCommand(_Cfg(tmp_path), None)._extract_pipe_tables(qmd)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.label == "tbl-cold-start"
    assert entry.source_file == "contents/vol2/inference/inference.qmd"
    assert entry.source_line == 8
    assert entry.columns == 2
    assert entry.rows == 2
    assert entry.has_colwidths is True
    assert entry.colwidths == "[60,40]"
    assert "Cold-Start Timeline" in entry.caption


def test_pipe_table_extraction_ignores_uncaptioned_tables(tmp_path):
    qmd_dir = tmp_path / "book" / "quarto" / "contents" / "vol2" / "ops"
    qmd_dir.mkdir(parents=True)
    qmd = qmd_dir / "ops.qmd"
    qmd.write_text(
        "| A | B |\n|---|---|\n| 1 | 2 |\n\nNo caption here.\n",
        encoding="utf-8",
    )

    assert LayoutCommand(_Cfg(tmp_path), None)._extract_pipe_tables(qmd) == []


def test_static_table_warnings_flag_dense_tables_not_simple_defaults():
    dense = LayoutCommand._static_table_warnings(
        columns=5,
        rows=6,
        max_cell_chars=48,
        has_colwidths=False,
    )
    simple = LayoutCommand._static_table_warnings(
        columns=2,
        rows=4,
        max_cell_chars=20,
        has_colwidths=True,
    )

    assert "wide-dense-table-without-colwidths" in dense
    assert "many-columns-without-colwidths" in dense
    assert "simple-table-has-colwidths-review-necessity" in simple
