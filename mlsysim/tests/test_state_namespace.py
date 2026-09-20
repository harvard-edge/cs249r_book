"""Volume isolation without reading or migrating a student's legacy ledger."""
import asyncio
import json
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from mlsysim.labs.state import DesignLedger


@pytest.mark.parametrize("volume", ["", "vol3", "../vol1", 'vol1";', 1, [], {}])
def test_invalid_volume_rejected_before_storage_access(monkeypatch, volume):
    def forbidden_home():
        raise AssertionError("invalid namespace accessed storage")
    monkeypatch.setattr(Path, "home", forbidden_home)
    with pytest.raises(ValueError, match="volume"):
        DesignLedger(volume=volume)


def test_native_same_chapter_isolated_and_legacy_not_migrated(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    DesignLedger().save(chapter=1, design={"choice": "legacy"})
    assert DesignLedger(volume="vol1").get_design(1) is None
    for volume in ("vol1", "vol2"):
        ledger = DesignLedger(volume=volume)
        ledger.save(chapter=1, design={"choice": volume})
        assert ledger.file_path == tmp_path / ".mlsys" / f"ledger_{volume}.json"
    for volume in ("vol1", "vol2"):
        assert DesignLedger(volume=volume).get_design(1) == {"choice": volume}
    assert DesignLedger().get_design(1) == {"choice": "legacy"}
    assert DesignLedger().file_path.name == "ledger.json"


def test_wasm_namespace_get_put_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(DesignLedger, "is_wasm", property(lambda self: True))
    store, calls = {}, []
    bridge = SimpleNamespace()
    js = ModuleType("js")
    js.globalThis = bridge
    code = ModuleType("pyodide.code")

    def run_js(script):
        calls.append(script)
        if "store.get(" in script:
            key = json.loads(re.search(r'store.get\(\s*("[^"]+")', script).group(1))
            value = store.get(key)
        else:
            key = json.loads(re.search(r'pendingState,\s*("[^"]+")', script).group(1))
            # Capture immediately, like the Promise executor before IndexedDB opens.
            assert "const pendingState = globalThis._mlsys_temp_state;" in script
            value = bridge._mlsys_temp_state
            async def put():
                await asyncio.sleep(0)
                store[key] = value
                return True
            return put()
        async def get():
            return value
        return get()

    code.run_js = run_js
    monkeypatch.setitem(sys.modules, "pyodide", ModuleType("pyodide"))
    monkeypatch.setitem(sys.modules, "pyodide.code", code)
    monkeypatch.setitem(sys.modules, "js", js)

    async def exercise():
        ledgers = [DesignLedger(volume=volume) for volume in (None, "vol1", "vol2")]
        await asyncio.gather(*(ledger.asave(chapter=1, design={"choice": ledger.volume}) for ledger in ledgers))
        for volume in (None, "vol1", "vol2"):
            reloaded = DesignLedger(volume=volume)
            await reloaded.load_async()
            assert reloaded.get_design(1) == {"choice": volume}
            assert reloaded.last_load_error is None
    asyncio.run(exercise())
    assert set(store) == {"mlsys_design_ledger", "mlsys_design_ledger_vol1", "mlsys_design_ledger_vol2"}
    assert len(calls) == 6
    assert all("__LEDGER_STORAGE_KEY__" not in script for script in calls)
