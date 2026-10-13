"""Schema v2 concept maps: the per-chapter ownership contract (added 2026-10-12)."""

from pathlib import Path

import yaml

from checks.concept_maps import check_concept_maps, render_volume_map


def _chapter(books: Path, folder: str, concept_map: dict) -> None:
    chapter = books / "vol1" / folder
    chapter.mkdir(parents=True)
    (chapter / f"{folder}.qmd").write_text(f"---\nconcepts: {folder}_concepts.yml\n---\n", encoding="utf-8")
    body = {"schema_version": 2, "source": f"{folder}.qmd", **concept_map}
    (chapter / f"{folder}_concepts.yml").write_text(yaml.safe_dump({"concept_map": body}), encoding="utf-8")


def _owns(*first: str, second: tuple[str, ...] = ()) -> dict:
    return {
        "primary_concepts": list(first),
        "secondary_concepts": list(second),
        "claims": {name: f"{name} claim." for name in (*first, *second)},
    }


def _messages(books: Path, root: Path, **kwargs) -> list[str]:
    return [f"{f.severity}: {f.message}" for f in check_concept_maps(books, root, **kwargs)]


def test_valid_volume_passes(tmp_path: Path) -> None:
    books = tmp_path / "books"
    _chapter(books, "01_intro", {**_owns("Iron law"), "previews": [{"concept": "Roofline", "owner": "02_hw"}]})
    _chapter(books, "02_hw", {**_owns("Roofline"), "touches": [{"concept": "Iron law", "owner": "01_intro"}]})
    assert _messages(books, tmp_path) == []


def test_missing_claim_and_stray_claim_are_errors(tmp_path: Path) -> None:
    books = tmp_path / "books"
    _chapter(books, "01_intro", {"primary_concepts": ["Iron law"], "secondary_concepts": [],
                                 "claims": {"Roofline": "Not owned here."}})
    messages = _messages(books, tmp_path)
    assert any("'Iron law' has no claim" in m for m in messages)
    assert any("claim for 'Roofline'" in m for m in messages)


def test_concept_owned_twice_is_an_error(tmp_path: Path) -> None:
    books = tmp_path / "books"
    _chapter(books, "01_intro", _owns("Iron law"))
    _chapter(books, "02_hw", _owns("Iron law"))
    assert any("a concept has one owner" in m for m in _messages(books, tmp_path))


def test_pointer_to_wrong_owner_is_an_error(tmp_path: Path) -> None:
    books = tmp_path / "books"
    _chapter(books, "01_intro", _owns("Iron law"))
    _chapter(books, "02_hw", {**_owns("Roofline"), "touches": [{"concept": "Iron law", "owner": "03_ops"}]})
    assert any("not a chapter folder" in m for m in _messages(books, tmp_path))


def test_preview_must_point_forward(tmp_path: Path) -> None:
    books = tmp_path / "books"
    _chapter(books, "01_intro", _owns("Iron law"))
    _chapter(books, "02_hw", {**_owns("Roofline"), "previews": [{"concept": "Iron law", "owner": "01_intro"}]})
    assert any("list it under touches" in m for m in _messages(books, tmp_path))


def test_forward_touch_only_reported_with_order(tmp_path: Path) -> None:
    books = tmp_path / "books"
    _chapter(books, "01_intro", {**_owns("Iron law"), "touches": [{"concept": "Roofline", "owner": "02_hw"}]})
    _chapter(books, "02_hw", _owns("Roofline"))
    assert _messages(books, tmp_path) == []
    assert any("used before it is taught" in m for m in _messages(books, tmp_path, include_order=True))


def test_too_many_first_order_concepts_warns(tmp_path: Path) -> None:
    books = tmp_path / "books"
    _chapter(books, "01_intro", _owns(*(f"C{i}" for i in range(9))))
    assert any("may be two chapters" in m for m in _messages(books, tmp_path))


def test_legacy_maps_skip_v2_rules(tmp_path: Path) -> None:
    books = tmp_path / "books"
    chapter = books / "vol2" / "01_intro"
    chapter.mkdir(parents=True)
    (chapter / "01_intro.qmd").write_text("---\nconcepts: 01_intro_concepts.yml\n---\n", encoding="utf-8")
    legacy = {"concept_map": {"source": "01_intro.qmd", "primary_concepts": ["A"], "secondary_concepts": ["B"]}}
    (chapter / "01_intro_concepts.yml").write_text(yaml.safe_dump(legacy), encoding="utf-8")
    assert _messages(books, tmp_path) == []


def test_render_volume_map_follows_config_order(tmp_path: Path) -> None:
    books = tmp_path / "books"
    _chapter(books, "01_intro", {**_owns("Iron law"), "previews": [{"concept": "Roofline", "owner": "02_hw"}]})
    _chapter(books, "02_hw", {**_owns("Roofline", second=("Ridge point",)),
                              "touches": [{"concept": "Iron law", "owner": "01_intro"}]})
    config = tmp_path / "_quarto-pdf-vol1.yml"
    config.write_text("    # Part I: Hardware\n    - vol1/02_hw/02_hw.qmd\n    - vol1/01_intro/01_intro.qmd\n",
                      encoding="utf-8")
    text = render_volume_map(books / "vol1", config)
    assert "2 chapters own 2 first-order and 1 second-order" in text
    assert text.index("### 02_hw") < text.index("### 01_intro")
    assert "## Part I: Hardware" in text
    assert "- **Roofline**: Roofline claim." in text
    assert "*Touches* (1): 01_intro: Iron law" in text
