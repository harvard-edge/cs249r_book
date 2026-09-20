"""Immutable experiment snapshots and workflow checks, without simulator math.

Only experimental settings belong in ``inputs``. Titles, colors, and other
presentation metadata belong outside that mapping and cannot establish contrast.
Snapshots accept JSON data; convert simulator quantities through its result API
before capture. Recapturing is explicit: replace a record in the caller's mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def input_fingerprint(*, track: str, inputs: Any) -> str:
    """Stable digest of a track and its upstream experimental dependencies."""
    return hashlib.sha256(_json({"track": track, "inputs": inputs}).encode()).hexdigest()


@dataclass(frozen=True)
class EvidenceCapture:
    """JSON-backed immutable record; exported dictionaries never alias it."""

    track: str
    part: str
    upstream_fingerprint: str
    _snapshot_json: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._snapshot_json)


def capture_evidence(
    *, track: str, part: str, prediction: Any, inputs: Mapping[str, Any],
    baseline: Any, result: Any, upstream_inputs: Any = None,
    alternatives: Any = (), decision: Any = None, model_key: str | None = None,
    chosen_result: Any = None, result_role: str = "tested intervention",
) -> EvidenceCapture:
    """Freeze an experiment and the student's prediction/choice at capture time.

    ``upstream_inputs`` contains dependencies that invalidate this record when
    changed, such as the preceding part's chosen design. It need not include the
    live experimental controls: moving a control should not erase saved evidence.
    For a within-part pair, baseline and result each contain an ``inputs`` mapping.
    """
    if not track or not part:
        raise ValueError("track and part must be nonempty")
    fingerprint = input_fingerprint(track=track, inputs=upstream_inputs)
    snapshot = _json(dict(
        track=track, part=part, prediction=prediction, inputs=inputs,
        baseline=baseline, result=result, alternatives=alternatives,
        decision=decision, model_key=model_key,
        chosen_result=chosen_result, result_role=result_role,
        upstream_fingerprint=fingerprint,
    ))
    return EvidenceCapture(track, part, fingerprint, snapshot)


@dataclass(frozen=True)
class EvidenceAudit:
    missing: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()
    identical_pairs: tuple[tuple[str, str], ...] = ()
    missing_contrasts: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not (self.missing or self.stale or self.identical_pairs or self.missing_contrasts)

    def to_dict(self) -> dict[str, Any]:
        return dict(complete=self.complete, missing=list(self.missing),
                    stale=list(self.stale),
                    identical_pairs=[list(pair) for pair in self.identical_pairs],
                    missing_contrasts=list(self.missing_contrasts))


def audit_evidence(
    captures: Mapping[str, EvidenceCapture], *, track: str,
    required_parts: Sequence[str], upstream_inputs: Any = None,
    contrast_pairs: Sequence[tuple[str, str]] = (),
    contrast_required_parts: Sequence[str] = (),
    per_part_upstream_inputs: Mapping[str, Any] | None = None,
) -> EvidenceAudit:
    """Check presence, dependency freshness, and actual input contrast.

    Does not judge predictions, numerical outputs, or engineering decisions.
    Missing per-part dependency entries fall back to ``upstream_inputs``.
    """
    required = tuple(dict.fromkeys((*required_parts, *contrast_required_parts,
                                   *(part for pair in contrast_pairs for part in pair))))
    missing = tuple(part for part in required if part not in captures)
    stale = []
    identical = []
    missing_contrasts = []
    snapshots = {}
    for part in required:
        if part not in captures:
            continue
        capture = captures[part]
        expected = (per_part_upstream_inputs or {}).get(part, upstream_inputs)
        if (capture.track != track or capture.part != part or
                capture.upstream_fingerprint != input_fingerprint(track=track, inputs=expected)):
            stale.append(part)
        snapshots[part] = capture.to_dict()
    for left, right in contrast_pairs:
        if left in snapshots and right in snapshots:
            if snapshots[left]["inputs"] == snapshots[right]["inputs"]:
                identical.append((left, right))
    for part in contrast_required_parts:
        if part not in snapshots:
            continue
        baseline, result = snapshots[part]["baseline"], snapshots[part]["result"]
        if (not isinstance(baseline, dict) or not isinstance(result, dict) or
                not isinstance(baseline.get("inputs"), dict) or
                not isinstance(result.get("inputs"), dict) or
                not baseline["inputs"] or not result["inputs"]):
            missing_contrasts.append(part)
        elif baseline["inputs"] == result["inputs"]:
            identical.append((part, part))
    return EvidenceAudit(missing, tuple(stale), tuple(identical), tuple(missing_contrasts))
