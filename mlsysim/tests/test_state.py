"""
Tests for DesignLedger persistence (mlsysim/labs/state.py).

Covers the WASM background-save failure path fixed in #1985: previously
`save()` used `asyncio.create_task(...)` fire-and-forget, so IndexedDB
failures inside `save_async()` were silently swallowed and never surfaced
to the caller. See:
https://github.com/harvard-edge/cs249r_book/issues/1985
"""

import asyncio

import pytest

import json
from pathlib import Path

from mlsysim.labs.state import DesignLedger, LedgerState
import mlsysim.labs.state as state_mod


def _ledger_with_home(monkeypatch, tmp_path):
    monkeypatch.setattr(state_mod.Path, "home", lambda: tmp_path)
    return DesignLedger()


def _force_wasm(monkeypatch, ledger):
    monkeypatch.setattr(type(ledger), "is_wasm", property(lambda self: True))


def test_save_and_load_roundtrip_native(tmp_path, monkeypatch):
    """Non-WASM path: save() should persist synchronously to disk."""
    ledger = _ledger_with_home(monkeypatch, tmp_path)
    ledger.save(track="edge", step=1, design={"foo": "bar"})

    reloaded = _ledger_with_home(monkeypatch, tmp_path)
    assert reloaded.get_track() == "edge"
    assert reloaded.get_design(1) == {"foo": "bar"}


def test_wasm_save_success_clears_error(tmp_path, monkeypatch):
    """A successful WASM background save should leave last_save_error unset."""
    ledger = _ledger_with_home(monkeypatch, tmp_path)
    _force_wasm(monkeypatch, ledger)

    async def fake_save_async(self):
        return True

    monkeypatch.setattr(DesignLedger, "save_async", fake_save_async)

    async def run():
        ledger.save(step=1, design={"ok": True})
        await ledger.flush()

    asyncio.run(run())
    assert ledger.last_save_error is None
    assert ledger.save_pending is False


def test_wasm_save_failure_is_captured_not_silent(tmp_path, monkeypatch):
    """Regression test for #1985.

    Previously: an IndexedDB failure inside save_async() was swallowed by
    a fire-and-forget asyncio.create_task(), so save() returned as if it
    had succeeded and students never learned their progress was lost.

    Now: the failure must be captured on `last_save_error`, and
    `save_pending` must go back to False once the background task settles.
    """
    ledger = _ledger_with_home(monkeypatch, tmp_path)
    _force_wasm(monkeypatch, ledger)

    async def failing_save_async(self):
        raise RuntimeError("indexedDB.open failed: QuotaExceededError")

    monkeypatch.setattr(DesignLedger, "save_async", failing_save_async)

    async def run():
        ledger.save(step=1, design={"will_fail": True})
        with pytest.raises(RuntimeError):
            await ledger.flush()

    asyncio.run(run())

    assert ledger.save_pending is False
    assert ledger.last_save_error is not None
    assert "QuotaExceededError" in ledger.last_save_error


def test_flush_observes_multiple_completed_saves(tmp_path, monkeypatch):
    """flush() must retain and observe every save, even after tasks finish."""
    ledger = _ledger_with_home(monkeypatch, tmp_path)
    _force_wasm(monkeypatch, ledger)
    completed = []

    async def fake_save_async(self):
        await asyncio.sleep(0)
        completed.append(True)

    monkeypatch.setattr(DesignLedger, "save_async", fake_save_async)

    async def run():
        first = ledger.save(step=1, design={"first": True})
        second = ledger.save(step=2, design={"second": True})
        await asyncio.gather(first, second)
        assert ledger.save_pending is False
        await ledger.flush()

    asyncio.run(run())
    assert len(completed) == 2
    assert ledger._pending_save_tasks == set()


def test_asave_raises_on_failure_directly(tmp_path, monkeypatch):
    """asave() should propagate the exception directly to an awaiting caller."""
    ledger = _ledger_with_home(monkeypatch, tmp_path)
    _force_wasm(monkeypatch, ledger)

    async def failing_save_async(self):
        raise RuntimeError("storage disabled")

    monkeypatch.setattr(DesignLedger, "save_async", failing_save_async)

    async def run():
        with pytest.raises(RuntimeError, match="storage disabled"):
            await ledger.asave(step=2, design={"x": 1})

    asyncio.run(run())


"""--- Read-path error handling (#1994) ---
Native/local-filesystem path only. WASM/IndexedDB is covered separately in labs/tests/test_wasm_persistence.py."""

def test_init_does_not_raise(tmp_path, monkeypatch):
    """Regression guard: last_load_error is a read-only @property backed
    by _last_load_error. Assigning self.last_load_error = ... anywhere
    (including __init__) raises AttributeError immediately, since the
    property has no setter. This is the exact bug that would otherwise
    only surface at runtime, not at review time."""
    _ledger_with_home(monkeypatch, tmp_path)  # must not raise


def test_load_missing_file_is_not_an_error(tmp_path, monkeypatch):
    """First run for a student -- no ledger.json exists yet. This is
        expected, not a failure, and must not populate last_load_error."""
    ledger = _ledger_with_home(monkeypatch, tmp_path)
    state = ledger.load()
    assert isinstance(state, LedgerState)
    assert ledger.last_load_error is None


def test_load_corrupt_file_sets_last_load_error(tmp_path, monkeypatch):
    """A save file exists but isn't valid JSON -- e.g. truncated by a
    crash mid-write. Must fail safe (blank state, no crash) but the
    failure must be visible via last_load_error, not silently discarded."""
    ledger = _ledger_with_home(monkeypatch, tmp_path)
    ledger.config_dir.mkdir(exist_ok=True)
    ledger.file_path.write_text("{not valid json")

    state = ledger.load()

    assert isinstance(state, LedgerState)
    assert ledger.last_load_error is not None
    assert "json" in ledger.last_load_error.lower()


def test_load_corrupt_file_resets_to_blank_state(tmp_path, monkeypatch):
    """A corrupt file must not leave stale/partial in-memory state around
    -- the fallback is a fresh LedgerState(), not a half-populated one."""
    ledger = _ledger_with_home(monkeypatch, tmp_path)
    ledger.config_dir.mkdir(exist_ok=True)
    ledger.file_path.write_text("{not valid json")

    state = ledger.load()

    assert state.track is None
    assert state.current_step == 0
    assert state.history == {}


def test_load_valid_file_clears_previous_error(tmp_path, monkeypatch):
    """last_load_error must reset on a subsequent successful load --
    it's a snapshot of the *most recent* attempt, not sticky forever."""
    ledger = _ledger_with_home(monkeypatch, tmp_path)
    ledger.config_dir.mkdir(exist_ok=True)
    ledger.file_path.write_text("{not valid json")
    ledger.load()
    assert ledger.last_load_error is not None

    ledger.file_path.write_text(json.dumps({
        "track": "edge",
        "current_step": 3,
        "history": {},
        "last_updated": "2026-08-10T00:00:00",
    }))
    state = ledger.load()

    assert ledger.last_load_error is None
    assert state.track == "edge"
    assert state.current_step == 3


def test_load_valid_file_round_trips_history(tmp_path, monkeypatch):
    """Sanity check that the happy path (already-existing behavior)
    wasn't broken by the error-handling changes."""
    ledger = _ledger_with_home(monkeypatch, tmp_path)
    ledger.config_dir.mkdir(exist_ok=True)
    ledger.file_path.write_text(json.dumps({
        "track": "cloud",
        "current_step": 5,
        "history": {"1": {"choice": "gpu"}, "5": {"choice": "spot"}},
        "last_updated": "2026-08-10T00:00:00",
    }))

    state = ledger.load()

    assert ledger.last_load_error is None
    assert state.history[1] == {"choice": "gpu"}
    assert state.history[5] == {"choice": "spot"}