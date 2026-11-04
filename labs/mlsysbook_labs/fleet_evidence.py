"""Deliberate persistence of completed fleet-lab discussions.

Report generation and downloads stay independent of persistence. Callers create
a fresh run button whenever discussion inputs change; this module checks the
exact payload before acknowledging a local save.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping


def discussion_fingerprint(design: Mapping[str, Any] | None) -> str:
    """Fingerprint the whole discussion, including immutable evidence and text."""
    encoded = json.dumps(design, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DiscussionSaveOutcome:
    status: str
    detail: str


async def save_completed_discussion(
    ledger: Any, *, chapter: int, track: str, design: Mapping[str, Any] | None,
    fingerprint: str, ready: bool, requested: bool,
) -> DiscussionSaveOutcome:
    """Persist only an explicit action; await browser storage before success."""
    if not ready or design is None:
        return DiscussionSaveOutcome(
            "Evidence in progress",
            "Complete current comparisons and the discussion before saving.",
        )
    if not requested:
        return DiscussionSaveOutcome(
            "Ready to save",
            "Download is available above. Select Save discussion to store this version locally; edits require saving again.",
        )
    if discussion_fingerprint(design) != fingerprint:
        return DiscussionSaveOutcome(
            "Discussion changed",
            "Save the current discussion again. Download remains available.",
        )
    # Copy JSON before awaiting storage so the record cannot alias live inputs.
    snapshot = json.loads(json.dumps(design, allow_nan=False))
    snapshot["discussion_fingerprint"] = fingerprint
    try:
        await ledger.load_async()
        await ledger.asave(track=track, chapter=chapter, design=snapshot)
    except Exception:
        return DiscussionSaveOutcome(
            "Storage unavailable",
            "The local save was not confirmed. Download the report above to keep this discussion, or retry Save discussion.",
        )
    return DiscussionSaveOutcome(
        "Saved locally",
        "This discussion version was saved on this device. Download remains available; edits require saving again.",
    )
