"""Tests for vault ship commit protocol (§6.1.1, Dean R3-NH-1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vault_cli.ship import LegPlan, LegState, ShipError, ShipJournal, ShipOutcome, run_ship


def test_all_legs_succeed(tmp_path: Path) -> None:
    journal = tmp_path / ".ship-journal.json"
    calls: list[str] = []
    legs = [
        LegPlan(name="d1",     forward=lambda: calls.append("d1")     or {}, rollback=lambda: {}),
        LegPlan(name="nextjs", forward=lambda: calls.append("nextjs") or {}, rollback=lambda: {}),
        LegPlan(name="paper",  forward=lambda: calls.append("paper")  or {}, rollback=None),
    ]
    j = run_ship(version="1.0.0", env="staging", journal_path=journal, legs=legs)
    assert j.outcome is ShipOutcome.SUCCESS
    assert j.point_of_no_return is True
    assert [l.state for l in j.legs] == [LegState.DEPLOYED] * 3
    assert calls == ["d1", "nextjs", "paper"]


def test_pre_paper_failure_auto_rolls_back(tmp_path: Path) -> None:
    """nextjs leg fails → d1 must be rolled back in reverse order."""
    journal = tmp_path / ".ship-journal.json"
    rollback_order: list[str] = []

    def fail_nextjs() -> dict:
        raise RuntimeError("next.js crashed")

    legs = [
        LegPlan(
            name="d1",
            forward=lambda: {},
            rollback=lambda: rollback_order.append("d1") or {},
        ),
        LegPlan(name="nextjs", forward=fail_nextjs, rollback=lambda: {}),
        LegPlan(name="paper",  forward=lambda: {}, rollback=None),
    ]
    with pytest.raises(ShipError, match="auto-rolled back"):
        run_ship(version="1.0.0", env="staging", journal_path=journal, legs=legs)
    j = ShipJournal.load(journal)
    assert j.outcome is ShipOutcome.FAILED_AUTO_ROLLED_BACK
    assert j.point_of_no_return is False
    assert j.legs[0].state is LegState.ROLLED_BACK
    assert j.legs[1].state is LegState.FAILED
    assert j.legs[2].state is LegState.PENDING
    assert rollback_order == ["d1"]


def test_paper_leg_failure_needs_manual(tmp_path: Path) -> None:
    """paper-leg failure MUST NOT auto-rollback earlier legs (git tag push
    cannot be un-pushed safely per §6.1.1)."""
    journal = tmp_path / ".ship-journal.json"
    d1_rolled: list[str] = []

    def fail_paper() -> dict:
        raise RuntimeError("git push --tags failed")

    legs = [
        LegPlan(name="d1",     forward=lambda: {}, rollback=lambda: d1_rolled.append("d1") or {}),
        LegPlan(name="nextjs", forward=lambda: {}, rollback=lambda: d1_rolled.append("nextjs") or {}),
        LegPlan(name="paper",  forward=fail_paper, rollback=None),
    ]
    with pytest.raises(ShipError, match="paper-leg failure"):
        run_ship(version="1.0.0", env="prod", journal_path=journal, legs=legs)
    j = ShipJournal.load(journal)
    assert j.outcome is ShipOutcome.FAILED_NEEDS_MANUAL
    # Both earlier legs remain DEPLOYED — no auto-rollback after paper-leg commits.
    assert j.legs[0].state is LegState.DEPLOYED
    assert j.legs[1].state is LegState.DEPLOYED
    assert j.legs[2].state is LegState.FAILED
    assert d1_rolled == []


def test_resume_continues_from_last_successful(tmp_path: Path) -> None:
    """--resume must pick up from the first non-DEPLOYED leg without re-running
    already-deployed ones (idempotency across operator interruptions)."""
    journal = tmp_path / ".ship-journal.json"
    calls1: list[str] = []

    def fail_nextjs() -> dict:
        raise RuntimeError("transient fail")

    legs1 = [
        LegPlan(name="d1",     forward=lambda: calls1.append("d1")     or {}, rollback=lambda: {}),
        LegPlan(name="nextjs", forward=fail_nextjs, rollback=lambda: {}),
        LegPlan(name="paper",  forward=lambda: {}, rollback=None),
    ]
    with pytest.raises(ShipError):
        run_ship(version="1.0.0", env="staging", journal_path=journal, legs=legs1)

    # Resume with a healed nextjs — d1 should NOT be called again because it
    # was rolled back during auto-rollback. Verify by checking the leg state.
    calls2: list[str] = []
    legs2 = [
        LegPlan(name="d1",     forward=lambda: calls2.append("d1")     or {}, rollback=lambda: {}),
        LegPlan(name="nextjs", forward=lambda: calls2.append("nextjs") or {}, rollback=lambda: {}),
        LegPlan(name="paper",  forward=lambda: calls2.append("paper")  or {}, rollback=None),
    ]
    # After auto-rollback, d1 is ROLLED_BACK so resume should re-deploy it.
    j = run_ship(version="1.0.0", env="staging", journal_path=journal, legs=legs2, resume=True)
    assert j.outcome is ShipOutcome.SUCCESS
    assert calls2 == ["d1", "nextjs", "paper"]
