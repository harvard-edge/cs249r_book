"""Tests for the legacy-JSON exporter (§11.1 migration contract)."""

from __future__ import annotations

import json
from pathlib import Path

from vault_cli.legacy_export import emit_legacy_corpus
from vault_cli.loader import LoadedQuestion
from vault_cli.models import (
    ChainRef,
    Details,
    Level,
    Provenance,
    Question,
    Status,
    Track,
    Zone,
)
from vault_cli.paths import Classification


def _make_lq(id: str, chain: ChainRef | None = None) -> LoadedQuestion:
    return LoadedQuestion(
        question=Question(
            id=id,
            title=f"T-{id}",
            topic="kv-cache-management",
            status=Status.published,
            provenance=Provenance.human,
            chain=chain,
            scenario="plaintext scenario.",
            details=Details(realistic_solution="answer."),
        ),
        classification=Classification(
            track=Track.cloud, level=Level.l4, zone=Zone.diagnosis
        ),
        path=Path(f"/tmp/{id}.yaml"),
    )


def test_legacy_shape_matches_site_interface(tmp_path: Path) -> None:
    """Emitted JSON items must carry every field the site's corpus.ts
    Question interface declares."""
    policy = tmp_path / "release-policy.yaml"
    policy.write_text("policy_version: 1\ninclude: {status: [published], require_validated: false}\n")
    out = tmp_path / "corpus.json"
    emit_legacy_corpus(tmp_path, [_make_lq("a"), _make_lq("b")], out)
    data = json.loads(out.read_text())
    assert len(data) == 2
    required = {
        "id", "track", "scope", "level", "title", "topic",
        "zone", "competency_area", "bloom_level", "scenario",
        "chain_ids", "chain_positions", "details",
    }
    for item in data:
        assert required.issubset(item.keys()), f"missing: {required - item.keys()}"


def test_chain_positions_legacy_shape(tmp_path: Path) -> None:
    """Legacy corpus.json used chain_positions as a {chain_id: 0-indexed} dict.
    Our new schema is 1-indexed; adapter must undo the +1."""
    policy = tmp_path / "release-policy.yaml"
    policy.write_text("policy_version: 1\ninclude: {status: [published], require_validated: false}\n")
    out = tmp_path / "corpus.json"
    chained = _make_lq("c", chain=ChainRef(id="my-chain", position=3))
    emit_legacy_corpus(tmp_path, [chained], out)
    data = json.loads(out.read_text())
    assert data[0]["chain_ids"] == ["my-chain"]
    # New schema position=3 → legacy {chain_id: 2} (0-indexed).
    assert data[0]["chain_positions"] == {"my-chain": 2}


def test_emitter_deterministic(tmp_path: Path) -> None:
    """Byte-stable output across repeat invocations — required for the CI
    equivalence check (§11.1)."""
    policy = tmp_path / "release-policy.yaml"
    policy.write_text("policy_version: 1\ninclude: {status: [published], require_validated: false}\n")
    out1 = tmp_path / "corpus1.json"
    out2 = tmp_path / "corpus2.json"
    lqs = [_make_lq(f"q-{i:03d}") for i in range(5)]
    # Intentionally reversed input order to verify sort.
    emit_legacy_corpus(tmp_path, list(reversed(lqs)), out1)
    emit_legacy_corpus(tmp_path, lqs, out2)
    assert out1.read_bytes() == out2.read_bytes()


def test_competency_area_resolves_to_canonical_area(tmp_path: Path) -> None:
    """competency_area must resolve to the canonical 13-area grouping from
    topics.json, NOT be a copy of the topic slug."""
    # Set up the topics.json in the expected relative path.
    topics_dir = tmp_path / "staffml" / "src" / "data"
    topics_dir.mkdir(parents=True)
    topics_data = [{"id": "kv-cache-management", "area": "memory"}]
    (topics_dir / "topics.json").write_text(json.dumps(topics_data))

    # vault_dir is tmp_path/vault so ../staffml resolves correctly.
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    policy = vault_dir / "release-policy.yaml"
    policy.write_text("policy_version: 1\ninclude: {status: [published], require_validated: false}\n")

    # Clear the module-level cache so our test topics.json is loaded.
    import vault_cli.legacy_export as _mod
    _mod._TOPIC_TO_AREA.clear()

    out = tmp_path / "corpus.json"
    emit_legacy_corpus(vault_dir, [_make_lq("a")], out)
    data = json.loads(out.read_text())
    assert data[0]["topic"] == "kv-cache-management"
    assert data[0]["competency_area"] == "memory"
    assert data[0]["scope"] == data[0]["track"]

    # Clean up cache for other tests.
    _mod._TOPIC_TO_AREA.clear()


def test_competency_area_falls_back_to_topic(tmp_path: Path) -> None:
    """When topics.json is absent, competency_area falls back to topic slug."""
    import vault_cli.legacy_export as _mod
    _mod._TOPIC_TO_AREA.clear()

    policy = tmp_path / "release-policy.yaml"
    policy.write_text("policy_version: 1\ninclude: {status: [published], require_validated: false}\n")
    out = tmp_path / "corpus.json"
    emit_legacy_corpus(tmp_path, [_make_lq("a")], out)
    data = json.loads(out.read_text())
    # No topics.json → fallback to topic slug.
    assert data[0]["competency_area"] == data[0]["topic"]

    _mod._TOPIC_TO_AREA.clear()
