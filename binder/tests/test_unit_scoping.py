"""Scoping guards for the numbers checks.

2026-09-20: `unit-spacing` and `binary-units` previously scanned raw prose
lines and skipped only fenced code blocks, so they fired inside inline math,
inline code spans, and pipe-table cells — the contexts book-prose.md exempts.
On Volume III that was 321 of 354 binary-unit hits and 17 of 20 unit-spacing
hits, every one a false positive. `currency` reported an SVG's XML comment as
a reader-facing defect. These tests pin both the exemptions and the genuine
detections so the masking cannot quietly regress.
"""

from pathlib import Path
from types import SimpleNamespace

from binder.cli.checks.currency_style import audit
from binder.cli.commands.validate import ValidateCommand


def _command(root: Path) -> ValidateCommand:
    return ValidateCommand(
        config_manager=SimpleNamespace(book_dir=root),
        chapter_discovery=None,
    )


def _issue_lines(root: Path, scope: str) -> set[int]:
    cmd = _command(root)
    runner = {
        "unit-spacing": cmd._run_unit_spacing,
        "binary-units": cmd._run_binary_units,
    }[scope]
    return {i.line for i in runner(root).issues}


def _write(root: Path, body: str) -> Path:
    chapter = root / "vol3/02_processor/02_processor.qmd"
    chapter.parent.mkdir(parents=True, exist_ok=True)
    chapter.write_text(body, encoding="utf-8")
    return chapter


def test_unit_spacing_exempts_math_code_tables_and_part_numbers(tmp_path):
    books = tmp_path / "books"
    _write(books, "\n".join([
        "Latency reached 100ms under load.",                 # 1 genuine
        "The budget is $4.5\\text{ GB}$ per replica.",        # 2 inline math
        "The bar prints `450MB/1000MB` to stderr.",           # 3 inline code
        "| Model | 80GB HBM |",                               # 4 table row
        "Provisioned on an NVIDIA A100-SXM4-80GB accelerator.",  # 5 part number
        "Serving a 70B parameter model.",                     # 6 size shorthand
    ]) + "\n")

    assert _issue_lines(books, "unit-spacing") == {1}


def test_binary_units_exempts_math_and_tables_but_flags_prose(tmp_path):
    books = tmp_path / "books"
    _write(books, "\n".join([
        "The host carries 512 GiB of DRAM.",                  # 1 genuine
        "$M_{KV} = 320\\text{ KiB}$ per token.",               # 2 inline math
        "| Tier | 40 GiB |",                                  # 3 table row
        "Set `--cache 8GiB` at startup.",                     # 4 inline code
    ]) + "\n")

    assert _issue_lines(books, "binary-units") == {1}


def test_binary_units_still_flags_every_binary_prefix(tmp_path):
    books = tmp_path / "books"
    _write(books, "Sizes of 4 KiB, 8 MiB, 16 GiB, and 2 TiB appear here.\n")

    assert _issue_lines(books, "binary-units") == {1}
    assert len(_command(books)._run_binary_units(books).issues) == 4


def test_currency_ignores_markup_comments_but_flags_visible_labels(tmp_path):
    books = tmp_path / "books"
    svg_dir = books / "vol3/17_tokenomics/images/svg"
    svg_dir.mkdir(parents=True)

    (svg_dir / "commented.svg").write_text(
        "<svg>\n  <!-- $ USD -->\n  <rect/>\n</svg>\n", encoding="utf-8"
    )
    assert audit([books]) == []

    (svg_dir / "labelled.svg").write_text(
        "<svg><text>Spend (USD)</text></svg>\n", encoding="utf-8"
    )
    assert [v.code for v in audit([books])] == ["currency_usd_literal"]


def test_currency_ignores_multiline_markup_comment(tmp_path):
    books = tmp_path / "books"
    svg_dir = books / "vol3/17_tokenomics/images/svg"
    svg_dir.mkdir(parents=True)
    (svg_dir / "block.svg").write_text(
        "<svg>\n<!-- generated\n     cost in USD\n-->\n<rect/>\n</svg>\n",
        encoding="utf-8",
    )

    assert audit([books]) == []
