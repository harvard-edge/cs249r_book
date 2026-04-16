"""Coordinator for ``vault ship`` — atomic release across D1 + Next.js + paper.

Implements ARCHITECTURE.md §6.1.1 commit protocol and closes Dean R3-NH-1.

Ordering (load-bearing, §6.1.1):
    1. D1 deploy               (rollback: R2 snapshot restore; always works)
    2. Next.js deploy          (rollback: wrangler pages rollback; idempotent)
    3. paper-tag push          (LAST; remote-durable, cannot be un-pushed)

After the paper-leg commits, ``point_of_no_return = true`` in the journal.
Past this point, auto-rollback is no longer safe; the only forward path is a
remediation release.

The code here is deliberately minimal — each leg calls out to the existing
primitive (``vault deploy``, a Next.js deploy hook, ``git push --tags``). This
keeps the contract explicit and unit-testable without requiring Cloudflare or
a live site.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable


class LegState(str, Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ShipOutcome(str, Enum):
    SUCCESS = "success"
    FAILED_AUTO_ROLLED_BACK = "failed_auto_rolled_back"
    FAILED_NEEDS_MANUAL = "failed_needs_manual"


@dataclass
class Leg:
    name: str
    state: LegState = LegState.PENDING
    started_at: str | None = None
    completed_at: str | None = None
    details: dict = field(default_factory=dict)


@dataclass
class ShipJournal:
    """On-disk journal at ``releases/<version>/.ship-journal.json``.

    Durable state across interruptions; ``--resume`` uses this to continue.
    """

    version: str
    env: str
    started_at: str
    legs: list[Leg] = field(default_factory=list)
    point_of_no_return: bool = False
    outcome: ShipOutcome | None = None

    def write(self, path: Path) -> None:
        payload = {
            "version": self.version,
            "env": self.env,
            "started_at": self.started_at,
            "legs": [
                {
                    "name": leg.name,
                    "state": leg.state.value,
                    "started_at": leg.started_at,
                    "completed_at": leg.completed_at,
                    "details": leg.details,
                }
                for leg in self.legs
            ],
            "point_of_no_return": self.point_of_no_return,
            "outcome": self.outcome.value if self.outcome else None,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ShipJournal":
        d = json.loads(path.read_text(encoding="utf-8"))
        j = cls(version=d["version"], env=d["env"], started_at=d["started_at"])
        for leg_d in d["legs"]:
            j.legs.append(
                Leg(
                    name=leg_d["name"],
                    state=LegState(leg_d["state"]),
                    started_at=leg_d.get("started_at"),
                    completed_at=leg_d.get("completed_at"),
                    details=leg_d.get("details") or {},
                )
            )
        j.point_of_no_return = d.get("point_of_no_return", False)
        out = d.get("outcome")
        j.outcome = ShipOutcome(out) if out else None
        return j


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class LegPlan:
    name: str
    forward: Callable[[], dict]
    rollback: Callable[[], dict] | None  # None = cannot auto-rollback (paper-leg)


class ShipError(Exception):
    """Raised when a ship leg fails and cannot be auto-rolled-back cleanly."""


def run_ship(
    *,
    version: str,
    env: str,
    journal_path: Path,
    legs: list[LegPlan],
    resume: bool = False,
) -> ShipJournal:
    """Execute the ship protocol with journaling and per-leg auto-rollback.

    Contract (§6.1.1):
    - Legs run in the order given; D1 first, Next.js second, paper LAST.
    - Any leg-failure BEFORE the paper-leg triggers auto-rollback of already-
      deployed legs (in reverse order).
    - If the paper-leg itself fails, we PAGE and stop — do not auto-rollback
      the earlier legs once we've advertised the release upstream.
    - ``--resume`` starts from the first non-DEPLOYED leg.
    """
    # Load or create journal. Hardened per Chip R4-C-3: a journal from a
    # different version MUST NOT be resumed against the current version —
    # that would publish a mixed-release state.
    if resume and journal_path.exists():
        j = ShipJournal.load(journal_path)
        if j.version != version:
            raise ShipError(
                f"journal at {journal_path} is for version {j.version!r}, not {version!r}. "
                f"Delete it explicitly if you want to restart, or invoke --resume "
                f"with the matching version."
            )
        if j.env != env:
            raise ShipError(
                f"journal env {j.env!r} doesn't match ship env {env!r}. "
                f"Refusing to cross environments."
            )
    else:
        if journal_path.exists() and not resume:
            raise ShipError(
                f"ship-journal already exists at {journal_path}; "
                f"pass --resume to continue or delete the file to restart."
            )
        j = ShipJournal(version=version, env=env, started_at=_now(),
                        legs=[Leg(name=p.name) for p in legs])
        j.write(journal_path)

    # Find the first leg to run (first not DEPLOYED).
    name_to_plan = {p.name: p for p in legs}
    completed_legs: list[Leg] = []

    for leg in j.legs:
        plan = name_to_plan.get(leg.name)
        if plan is None:
            continue

        if leg.state is LegState.DEPLOYED:
            completed_legs.append(leg)
            continue

        # Run the leg.
        leg.state = LegState.DEPLOYING
        leg.started_at = _now()
        j.write(journal_path)

        try:
            details = plan.forward()
            leg.state = LegState.DEPLOYED
            leg.completed_at = _now()
            leg.details.update(details or {})
            completed_legs.append(leg)
            j.write(journal_path)
        except Exception as exc:  # noqa: BLE001 — journal any failure
            leg.state = LegState.FAILED
            leg.details["error"] = str(exc)
            j.write(journal_path)

            is_paper_leg = (leg is j.legs[-1])
            if is_paper_leg:
                # Past point of no return — do NOT auto-rollback earlier legs.
                j.outcome = ShipOutcome.FAILED_NEEDS_MANUAL
                j.write(journal_path)
                raise ShipError(
                    f"paper-leg failure for {version}; remediate via forward-fix release. "
                    f"Earlier legs DEPLOYED; operator paging required."
                ) from exc

            # Pre-paper failure: auto-rollback in reverse order.
            for done in reversed(completed_legs):
                done_plan = name_to_plan.get(done.name)
                if not done_plan or done_plan.rollback is None:
                    continue
                try:
                    done_plan.rollback()
                    done.state = LegState.ROLLED_BACK
                    j.write(journal_path)
                except Exception as rb_exc:  # noqa: BLE001
                    done.details["rollback_error"] = str(rb_exc)
                    j.outcome = ShipOutcome.FAILED_NEEDS_MANUAL
                    j.write(journal_path)
                    raise ShipError(
                        f"auto-rollback of leg '{done.name}' failed: {rb_exc}"
                    ) from rb_exc
            j.outcome = ShipOutcome.FAILED_AUTO_ROLLED_BACK
            j.write(journal_path)
            raise ShipError(f"leg '{leg.name}' failed; auto-rolled back") from exc

    # After paper-leg: point of no return.
    j.point_of_no_return = True
    j.outcome = ShipOutcome.SUCCESS
    j.write(journal_path)
    return j


__all__ = ["LegPlan", "LegState", "ShipError", "ShipJournal", "ShipOutcome", "run_ship"]
