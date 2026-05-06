"""Smoke tests for the audit_corpus_batched batching helper.

Verifies that pack_batches:
  - preserves every input item across batches (no dropped items)
  - preserves input order within and across batches
  - respects max_chars (batch payload character total stays within budget)
  - respects max_items_per_batch (hard cap)
  - handles empty input
  - emits a single batch when input fits

The audit script itself can't easily be unit-tested in CI (it
subprocess-shells the gemini CLI); the batching helper is the main
piece of pure logic, so it's where the test value is.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# scripts/ is not on the standard path; insert ad-hoc.
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from _batching import pack_batches  # noqa: E402


def _payload(d: dict) -> dict:
    return d


def test_empty_input_returns_no_batches() -> None:
    assert pack_batches([], payload_for=_payload) == []


def test_single_small_item_one_batch() -> None:
    items = [{"id": "x"}]
    batches = pack_batches(items, payload_for=_payload)
    assert len(batches) == 1
    assert batches[0] == items


def test_no_items_lost_across_batches() -> None:
    items = [{"id": f"q{i}", "body": "x" * 100} for i in range(50)]
    batches = pack_batches(items, payload_for=_payload, max_items_per_batch=10)
    flat = [it for batch in batches for it in batch]
    assert flat == items
    assert len(flat) == 50


def test_max_items_per_batch_caps_batch_size() -> None:
    items = [{"id": f"q{i}"} for i in range(33)]
    batches = pack_batches(items, payload_for=_payload, max_items_per_batch=10)
    sizes = [len(b) for b in batches]
    assert sizes == [10, 10, 10, 3]


def test_max_chars_triggers_batch_flush() -> None:
    big_item_body = "x" * 1_000
    items = [{"id": f"q{i}", "body": big_item_body} for i in range(20)]
    # Wrapper + ~5 items per batch should fit; 6 should not.
    batches = pack_batches(
        items,
        payload_for=_payload,
        max_chars=8_000,
        wrapper_chars=1_000,
    )
    # No batch should exceed the budget when serialized.
    for b in batches:
        total = 1_000 + sum(len(json.dumps(_payload(x))) for x in b)
        assert total <= 8_000 + len(json.dumps(_payload(b[-1])))  # last item may push over
    # Every item is accounted for.
    flat = [it for batch in batches for it in batch]
    assert flat == items


def test_input_order_preserved() -> None:
    items = [{"id": f"q{i:03d}"} for i in range(25)]
    batches = pack_batches(items, payload_for=_payload, max_items_per_batch=7)
    flat = [it for batch in batches for it in batch]
    assert [x["id"] for x in flat] == [x["id"] for x in items]


def test_single_oversized_item_still_lands_in_a_batch() -> None:
    """If one item alone exceeds max_chars, we don't drop it — we let
    the prompt overflow rather than silently lose data. The caller is
    expected to detect overflow downstream (e.g., parse failure)."""
    huge = {"id": "huge", "body": "x" * 100_000}
    small = {"id": "small"}
    batches = pack_batches(
        [huge, small],
        payload_for=_payload,
        max_chars=10_000,
        wrapper_chars=500,
    )
    flat = [it for batch in batches for it in batch]
    assert flat == [huge, small]
