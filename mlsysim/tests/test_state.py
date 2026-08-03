"""Tests for DesignLedger persistence (mlsysim/labs/state.py).

Covers the WASM background-save failure path fixed in #1985: previously
`save()` used `asyncio.create_task(...)` fire-and-forget, so IndexedDB
failures inside `save_async()` were silently swallowed and never surfaced
to the caller. See:
https://github.com/harvard-edge/cs249r_book/issues/1985
"""
import asyncio

import pytest

from mlsysim.labs.state import DesignLedger
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
