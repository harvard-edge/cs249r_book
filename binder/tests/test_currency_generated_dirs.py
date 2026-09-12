"""The currency check must not scan render caches left by a local build."""

from pathlib import Path

from checks.currency_style import iter_target_files


def test_render_caches_are_not_scanned(tmp_path: Path) -> None:
    chapter = tmp_path / "vol1" / "01_intro" / "01_intro.qmd"
    chapter.parent.mkdir(parents=True)
    chapter.write_text("Costs are $5.\n", encoding="utf-8")

    cache = tmp_path / ".quarto" / "idx" / "vol1" / "01_intro.qmd.json"
    cache.parent.mkdir(parents=True)
    cache.write_text('{"markdown": "5 USD"}\n', encoding="utf-8")

    figures = tmp_path / "vol1" / "01_intro" / "01_intro_files" / "note.qmd"
    figures.parent.mkdir(parents=True)
    figures.write_text("5 USD\n", encoding="utf-8")

    found = {p.relative_to(tmp_path).as_posix() for p in iter_target_files([tmp_path])}

    assert "vol1/01_intro/01_intro.qmd" in found
    assert not any(part.startswith(".quarto") for part in found)
    assert not any("_files/" in part for part in found)
