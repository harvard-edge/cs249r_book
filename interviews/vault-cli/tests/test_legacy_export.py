"""Tests for the legacy-JSON exporter (v1.0)."""

from __future__ import annotations

import json
from pathlib import Path

from vault_cli.legacy_export import emit_legacy_corpus
from vault_cli.loader import LoadedQuestion
from vault_cli.models import (
    ChainRef,
    Details,
    Question,
)


def _make_lq(
    id: str,
    chains: list[ChainRef] | None = None,
    topic: str = "kv-cache-management",
    competency_area: str = "memory",
) -> LoadedQuestion:
    return LoadedQuestion(
        question=Question(
            id=id,
            track="cloud",
            level="L4",
            zone="diagnosis",
            topic=topic,
            competency_area=competency_area,
            bloom_level="analyze",
            title=f"T-{id}",
            scenario="plaintext scenario that is long enough to be useful.",
            details=Details(realistic_solution="answer."),
            status="published",
            provenance="human",
            chains=chains,
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
        "id", "track", "level", "title", "topic",
        "zone", "competency_area", "bloom_level", "scenario",
        "details",
    }
    for item in data:
        assert required.issubset(item.keys()), f"missing: {required - item.keys()}"
    # v1.0: dropped legacy `scope` field.
    for item in data:
        assert "scope" not in item, "scope was retired in v1.0"


def test_chain_positions_plural_preserved(tmp_path: Path) -> None:
    """v1.0 schema uses plural chains and preserves position verbatim."""
    policy = tmp_path / "release-policy.yaml"
    policy.write_text("policy_version: 1\ninclude: {status: [published], require_validated: false}\n")
    out = tmp_path / "corpus.json"
    chained = _make_lq("c", chains=[ChainRef(id="my-chain", position=3)])
    emit_legacy_corpus(tmp_path, [chained], out)
    data = json.loads(out.read_text())
    assert data[0]["chain_ids"] == ["my-chain"]
    assert data[0]["chain_positions"] == {"my-chain": 3}


def test_emitter_deterministic(tmp_path: Path) -> None:
    """Byte-stable output across repeat invocations — required for the CI
    equivalence check."""
    policy = tmp_path / "release-policy.yaml"
    policy.write_text("policy_version: 1\ninclude: {status: [published], require_validated: false}\n")
    out1 = tmp_path / "corpus1.json"
    out2 = tmp_path / "corpus2.json"
    lqs = [_make_lq(f"q-{i:03d}") for i in range(5)]
    # Intentionally reversed input order to verify sort.
    emit_legacy_corpus(tmp_path, list(reversed(lqs)), out1)
    emit_legacy_corpus(tmp_path, lqs, out2)
    assert out1.read_bytes() == out2.read_bytes()


def test_competency_area_preserved(tmp_path: Path) -> None:
    """competency_area is now a YAML field on the question; the exporter
    passes it through verbatim (no more topic→area lookup)."""
    policy = tmp_path / "release-policy.yaml"
    policy.write_text("policy_version: 1\ninclude: {status: [published], require_validated: false}\n")
    out = tmp_path / "corpus.json"
    emit_legacy_corpus(
        tmp_path,
        [_make_lq("a", topic="kv-cache-management", competency_area="memory")],
        out,
    )
    data = json.loads(out.read_text())
    assert data[0]["topic"] == "kv-cache-management"
    assert data[0]["competency_area"] == "memory"


def test_multi_chain_membership(tmp_path: Path) -> None:
    """v1.0 fix: a question belonging to multiple chains must surface all of
    them in chain_ids/chain_positions — v0.1 silently dropped all but one."""
    policy = tmp_path / "release-policy.yaml"
    policy.write_text("policy_version: 1\ninclude: {status: [published], require_validated: false}\n")
    out = tmp_path / "corpus.json"
    q = _make_lq(
        "multi",
        chains=[
            ChainRef(id="chain-a", position=1),
            ChainRef(id="chain-b", position=0),
            ChainRef(id="chain-c", position=2),
        ],
    )
    emit_legacy_corpus(tmp_path, [q], out)
    data = json.loads(out.read_text())
    assert set(data[0]["chain_ids"]) == {"chain-a", "chain-b", "chain-c"}
    assert data[0]["chain_positions"] == {
        "chain-a": 1, "chain-b": 0, "chain-c": 2,
    }
