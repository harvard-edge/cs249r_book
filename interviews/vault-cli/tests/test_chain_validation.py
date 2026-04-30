"""Tests for ``validate_chain`` in ``scripts/build_chains_with_gemini.py``.

Phase 1.3 of CHAIN_ROADMAP.md added a ``mode`` parameter that toggles the
allowed Bloom-level deltas:

    strict  → Δ ∈ {1, 2}
    lenient → Δ ∈ {0, 1, 2, 3}

These tests pin both directions: that lenient mode accepts shapes strict
mode rejects (Δ=0 same-scenario pairs, Δ=3 missing-rung jumps) and that
both modes still reject backward deltas, multi-topic chains, and
out-of-range chain sizes.

The script lives outside the importable ``vault_cli`` package, so we load
it via ``importlib.util`` rather than a normal import.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_chains_with_gemini.py"
)


@pytest.fixture(scope="module")
def chain_module():
    spec = importlib.util.spec_from_file_location("_build_chains", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def corpus():
    """Mini corpus: same topic at L1..L6+, plus one Δ=0 pair and one off-topic."""
    return {
        "x-1": {"level": "L1", "topic": "t", "track": "edge"},
        "x-2": {"level": "L2", "topic": "t", "track": "edge"},
        "x-3a": {"level": "L3", "topic": "t", "track": "edge"},
        "x-3b": {"level": "L3", "topic": "t", "track": "edge"},
        "x-4": {"level": "L4", "topic": "t", "track": "edge"},
        "x-5": {"level": "L5", "topic": "t", "track": "edge"},
        "x-6": {"level": "L6+", "topic": "t", "track": "edge"},
        "y-2": {"level": "L2", "topic": "u", "track": "edge"},
    }


@pytest.fixture
def bucket(corpus):
    return set(corpus.keys())


def _chain(*qids):
    return {"questions": list(qids)}


# --- strict mode --------------------------------------------------------

def test_strict_accepts_plus_one_progression(chain_module, bucket, corpus):
    ok, why = chain_module.validate_chain(
        _chain("x-1", "x-2", "x-3a"), bucket, corpus, mode="strict"
    )
    assert ok, why


def test_strict_accepts_plus_two_jump(chain_module, bucket, corpus):
    ok, why = chain_module.validate_chain(
        _chain("x-1", "x-3a"), bucket, corpus, mode="strict"
    )
    assert ok, why


def test_strict_rejects_same_level_pair(chain_module, bucket, corpus):
    ok, why = chain_module.validate_chain(
        _chain("x-3a", "x-3b"), bucket, corpus, mode="strict"
    )
    assert not ok
    assert "Δ=" in why


def test_strict_rejects_three_step_jump(chain_module, bucket, corpus):
    ok, why = chain_module.validate_chain(
        _chain("x-3a", "x-6"), bucket, corpus, mode="strict"
    )
    assert not ok
    assert "Δ=" in why


def test_strict_rejects_backward_step(chain_module, bucket, corpus):
    ok, why = chain_module.validate_chain(
        _chain("x-2", "x-1"), bucket, corpus, mode="strict"
    )
    assert not ok


# --- lenient mode -------------------------------------------------------

def test_lenient_accepts_same_level_pair(chain_module, bucket, corpus):
    """Δ=0 is allowed under lenient (shared-scenario assumption per prompt)."""
    ok, why = chain_module.validate_chain(
        _chain("x-3a", "x-3b"), bucket, corpus, mode="lenient"
    )
    assert ok, why


def test_lenient_accepts_three_step_jump(chain_module, bucket, corpus):
    """Δ=3 is allowed under lenient when no smaller intermediate exists."""
    ok, why = chain_module.validate_chain(
        _chain("x-3a", "x-6"), bucket, corpus, mode="lenient"
    )
    assert ok, why


def test_lenient_accepts_mixed_long_chain(chain_module, bucket, corpus):
    ok, why = chain_module.validate_chain(
        _chain("x-1", "x-2", "x-3a", "x-5", "x-6"),
        bucket, corpus, mode="lenient",
    )
    assert ok, why


def test_lenient_still_rejects_backward(chain_module, bucket, corpus):
    ok, why = chain_module.validate_chain(
        _chain("x-2", "x-1"), bucket, corpus, mode="lenient"
    )
    assert not ok


def test_lenient_rejects_four_step_jump(chain_module, bucket, corpus):
    """Δ=4 (e.g., L1→L5) stays out of bounds even under lenient."""
    ok, why = chain_module.validate_chain(
        _chain("x-1", "x-5"), bucket, corpus, mode="lenient"
    )
    assert not ok


# --- both modes ---------------------------------------------------------

@pytest.mark.parametrize("mode", ["strict", "lenient"])
def test_size_below_two_rejected(chain_module, bucket, corpus, mode):
    ok, why = chain_module.validate_chain(
        _chain("x-1"), bucket, corpus, mode=mode
    )
    assert not ok
    assert "size" in why


@pytest.mark.parametrize("mode", ["strict", "lenient"])
def test_size_above_six_rejected(chain_module, bucket, corpus, mode):
    ok, why = chain_module.validate_chain(
        _chain("x-1", "x-2", "x-3a", "x-3b", "x-4", "x-5", "x-6"),
        bucket, corpus, mode=mode,
    )
    assert not ok
    assert "size" in why


@pytest.mark.parametrize("mode", ["strict", "lenient"])
def test_multi_topic_rejected(chain_module, bucket, corpus, mode):
    ok, why = chain_module.validate_chain(
        _chain("x-1", "y-2"), bucket, corpus, mode=mode
    )
    assert not ok
    assert "multi-topic" in why


@pytest.mark.parametrize("mode", ["strict", "lenient"])
def test_unknown_qid_rejected(chain_module, bucket, corpus, mode):
    ok, why = chain_module.validate_chain(
        _chain("x-1", "x-99"), bucket, corpus, mode=mode
    )
    assert not ok
    assert "not in bucket" in why


def test_unknown_mode_rejected(chain_module, bucket, corpus):
    ok, why = chain_module.validate_chain(
        _chain("x-1", "x-2"), bucket, corpus, mode="moderate"
    )
    assert not ok
    assert "unknown mode" in why
