"""Shared batching helper for Gemini-CLI prompts.

Generalized from audit_chains_with_gemini.py:batch_chains and
build_chains_with_gemini.py:plan_batches. Pack a list of items into
batches whose serialized JSON payload stays under MAX_PROMPT_CHARS,
leaving wrapper room for the prompt scaffolding.

Used by:
  - audit_corpus_batched.py (CORPUS_HARDENING_PLAN.md Phase 3)
  - eventual rewrite of audit_chains_with_gemini.py (out of scope here)

Tuning:
  - MAX_PROMPT_CHARS = 320,000 chars ≈ 80,000 tokens at the typical
    English ratio. This is the "attention sweet spot" for
    gemini-3.1-pro-preview: large enough to amortize call overhead,
    small enough that the model still attends to every payload item.
  - DEFAULT_WRAPPER_CHARS = 4,000 chars. Empirically enough headroom
    for the prompt instructions, JSON schema description, and any
    in-prompt context (e.g., the FAILURE_MODE_TAXONOMY block).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

# Public tuning constants — callers may pass smaller limits via the
# `max_chars` parameter for tighter batches (e.g., when the per-item
# response is bigger and headroom for output matters).
MAX_PROMPT_CHARS = 320_000
DEFAULT_WRAPPER_CHARS = 4_000


def pack_batches[T](
    items: Sequence[T],
    *,
    payload_for: Callable[[T], object],
    max_chars: int = MAX_PROMPT_CHARS,
    wrapper_chars: int = DEFAULT_WRAPPER_CHARS,
    max_items_per_batch: int | None = None,
) -> list[list[T]]:
    """Pack ``items`` into batches that fit within ``max_chars`` total.

    The character budget is computed against the JSON-serialized
    payload of each item (``payload_for(item)``), plus the wrapper
    overhead. Items are NOT reordered; the input order is preserved
    within and across batches so callers get deterministic batching.

    Args:
        items: input sequence to batch.
        payload_for: function returning a JSON-serializable
            representation of one item (the prompt sees this, not the
            full Python object).
        max_chars: upper bound on total prompt characters per batch.
            Default 320K (= ~80K tokens).
        wrapper_chars: scaffolding overhead per batch (instructions,
            schema, in-prompt context). Default 4K.
        max_items_per_batch: optional hard cap on items-per-batch
            regardless of character budget. Useful when per-item
            output volume (not input volume) is the limiting factor.

    Returns:
        list of batches. Each batch is a list of items in original order.

    Properties:
      - ``sum(len(b) for b in pack_batches(items, ...)) == len(items)``
        (every item lands in exactly one batch)
      - input ordering preserved
      - empty ``items`` returns ``[]``
      - an item whose payload alone exceeds ``max_chars - wrapper_chars``
        still ends up in its own batch (we don't drop oversized items;
        the prompt will likely overflow but that's the caller's problem
        to detect downstream)
    """
    batches: list[list[T]] = []
    current: list[T] = []
    current_chars = wrapper_chars

    for item in items:
        item_chars = len(json.dumps(payload_for(item)))

        flush_for_chars = (
            current and current_chars + item_chars > max_chars
        )
        flush_for_count = (
            max_items_per_batch is not None
            and len(current) >= max_items_per_batch
        )

        if flush_for_chars or flush_for_count:
            batches.append(current)
            current = []
            current_chars = wrapper_chars

        current.append(item)
        current_chars += item_chars

    if current:
        batches.append(current)

    return batches


__all__ = [
    "MAX_PROMPT_CHARS",
    "DEFAULT_WRAPPER_CHARS",
    "pack_batches",
]
