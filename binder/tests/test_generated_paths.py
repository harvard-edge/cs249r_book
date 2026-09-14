"""Source checks must ignore render output and caches under books/."""

from pathlib import Path

from checks.concept_maps import check_concept_maps
from checks.generated_paths import is_generated


def test_is_generated_flags_build_trees(tmp_path: Path) -> None:
    assert is_generated(tmp_path / "_build" / "html-vol1" / "index.qmd", tmp_path)
    assert is_generated(tmp_path / ".quarto" / "idx" / "a.json", tmp_path)
    assert is_generated(tmp_path / "vol1" / "01_intro" / "01_intro_files" / "f.png", tmp_path)
    assert not is_generated(tmp_path / "vol1" / "01_intro" / "01_intro.qmd", tmp_path)


def test_concept_maps_ignore_stale_build_copies(tmp_path: Path) -> None:
    books = tmp_path / "books"
    stale = books / "_build" / "html-vol1" / "vol1" / "conclusion" / "conclusion.qmd"
    stale.parent.mkdir(parents=True)
    stale.write_text("---\nconcepts: conclusion_concepts.yml\n---\n", encoding="utf-8")

    findings = check_concept_maps(books, tmp_path)

    assert not [f for f in findings if "_build" in f.path]
