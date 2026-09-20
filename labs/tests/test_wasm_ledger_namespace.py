"""Real IndexedDB volume isolation, including overlapping async writes."""
import json

from .test_wasm_persistence import served_dir  # shared isolated Pyodide page fixture


def test_wasm_volumes_survive_concurrent_saves(served_dir):
    from playwright.sync_api import sync_playwright

    _, port = served_dir
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{port}/")
            page.wait_for_function("window.__ready || window.__initError", timeout=60000)
            assert page.evaluate("window.__initError || null") is None
            raw = page.evaluate('''async () => await window.__pyodide.runPythonAsync(`
import asyncio, json
async def namespace_roundtrip():
    ledgers = [DesignLedger(volume=v) for v in (None, "vol1", "vol2")]
    await asyncio.gather(*(ledger.asave(chapter=1, design={"choice": ledger.volume}) for ledger in ledgers))
    results = {}
    for volume in (None, "vol1", "vol2"):
        fresh = DesignLedger(volume=volume)
        await fresh.load_async()
        results[str(volume)] = {"design": fresh.get_design(1), "error": fresh.last_load_error}
    return json.dumps(results)
await namespace_roundtrip()
`)''')
            assert json.loads(raw) == {
                str(volume): {"design": {"choice": volume}, "error": None}
                for volume in (None, "vol1", "vol2")
            }
        finally:
            browser.close()
