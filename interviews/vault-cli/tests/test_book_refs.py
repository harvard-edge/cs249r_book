"""Tests for the topic → textbook chapter resolver and its link-checker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vault_cli.book_refs import BookRefError, BookRefResolver
from vault_cli.legacy_export import emit_corpus_summary
from vault_cli.loader import LoadedQuestion
from vault_cli.models import Details, Question


def _vault_with_map(tmp_path: Path, mapping: str) -> Path:
    """Lay out a minimal vault dir whose repo-root has a book/ tree.

    vault_dir = <root>/interviews/vault so that vault_dir.parent.parent is the
    repo root the resolver derives the book/quarto/contents path from.
    """
    vault_dir = tmp_path / "interviews" / "vault"
    (vault_dir / "schema").mkdir(parents=True)
    (vault_dir / "schema" / "topic_chapter_map.yaml").write_text(mapping, encoding="utf-8")
    return vault_dir


def _write_chapter(vault_dir: Path, vol: int, chapter: str, h1: str) -> None:
    root = vault_dir.parent.parent / "book" / "quarto" / "contents" / f"vol{vol}" / chapter
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{chapter}.qmd").write_text(f"{h1}\n\nbody\n", encoding="utf-8")


def test_resolves_primary_and_also_see_with_qmd_title(tmp_path: Path) -> None:
    vault_dir = _vault_with_map(
        tmp_path,
        "attention-scaling:\n"
        "  primary:  { vol: 1, chapter: nn_architectures }\n"
        "  also_see: [ { vol: 2, chapter: inference } ]\n"
        '  why: "MHA/GQA variants are architectural."\n',
    )
    _write_chapter(vault_dir, 1, "nn_architectures", "# Neural Architectures {#sec-nn-arch}")
    _write_chapter(vault_dir, 2, "inference", "# Model Inference")

    refs = BookRefResolver(vault_dir).refs_for_topic("attention-scaling")

    assert refs == [
        {
            "vol": 1,
            "chapter": "nn_architectures",
            "title": "Neural Architectures",  # anchor stripped, from .qmd H1
            "url": "https://mlsysbook.ai/vol1/contents/vol1/nn_architectures/nn_architectures.html",
            "role": "primary",
            "why": "MHA/GQA variants are architectural.",
        },
        {
            "vol": 2,
            "chapter": "inference",
            "title": "Model Inference",
            "url": "https://mlsysbook.ai/vol2/contents/vol2/inference/inference.html",
            "role": "also_see",  # also_see carries no `why`
        },
    ]


def test_unmapped_topic_returns_empty(tmp_path: Path) -> None:
    vault_dir = _vault_with_map(
        tmp_path,
        "attention-scaling:\n  primary: { vol: 1, chapter: nn_architectures }\n",
    )
    _write_chapter(vault_dir, 1, "nn_architectures", "# Neural Architectures")
    assert BookRefResolver(vault_dir).refs_for_topic("totally-unknown-topic") == []


def test_link_check_fails_on_missing_chapter(tmp_path: Path) -> None:
    """A mapped chapter with no .qmd source must fail the build."""
    vault_dir = _vault_with_map(
        tmp_path,
        "attention-scaling:\n  primary: { vol: 1, chapter: nonexistent_chapter }\n",
    )
    # Create the book tree (so link-check is active) but NOT the chapter.
    (vault_dir.parent.parent / "book" / "quarto" / "contents").mkdir(parents=True)

    with pytest.raises(BookRefError, match="nonexistent_chapter"):
        BookRefResolver(vault_dir)


def test_link_check_skipped_and_slug_fallback_when_no_book_tree(tmp_path: Path) -> None:
    """Standalone (no book tree): no link-check, titles fall back to slug."""
    vault_dir = _vault_with_map(
        tmp_path,
        "attention-scaling:\n  primary: { vol: 1, chapter: hw_acceleration }\n",
    )
    # No book/ tree created → link-check skipped, must not raise.
    refs = BookRefResolver(vault_dir).refs_for_topic("attention-scaling")
    assert refs[0]["title"] == "Hw Acceleration"  # slug title-case fallback


def test_absent_map_is_noop(tmp_path: Path) -> None:
    """No map file → resolver yields nothing (existing builds unaffected)."""
    vault_dir = tmp_path / "interviews" / "vault"
    vault_dir.mkdir(parents=True)
    assert BookRefResolver(vault_dir).refs_for_topic("anything") == []


def _make_lq(topic: str) -> LoadedQuestion:
    return LoadedQuestion(
        question=Question(
            id="cloud-0001",
            track="cloud",
            level="L4",
            zone="diagnosis",
            topic=topic,
            competency_area="memory",
            bloom_level="analyze",
            title="T",
            scenario="plaintext scenario that is long enough to be useful.",
            details=Details(realistic_solution="answer."),
            status="published",
            provenance="human",
        ),
        path=Path("/tmp/cloud-0001.yaml"),
    )


def test_book_refs_survive_into_summary_bundle(tmp_path: Path) -> None:
    """book_refs is top-level so it survives _to_summary (which drops only
    scenario + details) and renders synchronously from the summary bundle."""
    vault_dir = _vault_with_map(
        tmp_path,
        "kv-cache-management:\n  primary: { vol: 2, chapter: inference }\n"
        '  why: "KV cache lives in the inference chapter."\n',
    )
    _write_chapter(vault_dir, 2, "inference", "# Model Inference")
    (vault_dir / "release-policy.yaml").write_text(
        "policy_version: 1\ninclude: {status: [published], require_validated: false}\n"
    )

    out = vault_dir / "corpus-summary.json"
    emit_corpus_summary(vault_dir, [_make_lq("kv-cache-management")], out)
    item = json.loads(out.read_text(encoding="utf-8"))[0]

    assert item["scenario"] == ""  # heavy field stubbed...
    assert item["book_refs"][0]["chapter"] == "inference"  # ...but book_refs kept
    assert item["book_refs"][0]["title"] == "Model Inference"
