"""
Real-browser regression test for DesignLedger's WASM/IndexedDB persistence
(mlsysim/mlsysim/labs/state.py).

Why this exists
----------------
#1985 was a fire-and-forget asyncio.create_task() bug in DesignLedger.save()
that swallowed IndexedDB failures silently. PR #1988 fixed that -- but
review of #1988 caught a *second*, more insidious bug that mocked unit
tests (tests/test_state.py) could never catch: a Python name-mangling
issue. `globalThis.__mlsys_temp_state`, written inside the DesignLedger
class body, was silently rewritten by the Python compiler to
`globalThis._DesignLedger__mlsys_temp_state`, desyncing it from the plain
`__mlsys_temp_state` the embedded JS string read. The result: save_async()
reported success on every call while actually persisting `undefined` --
deterministically, only when invoked as a bound DesignLedger method.

Mocked tests can't catch this class of bug because they replace
save_async() entirely, so they only verify the bookkeeping logic (done
callbacks, last_save_error, flush(), asave() exception propagation) built
around whatever save_async() reports -- never whether save_async() itself
tells the truth against real IndexedDB.

This test runs the REAL, unmodified save_async() against a REAL Pyodide
runtime and REAL IndexedDB in headless Chromium, then reads the data back
out through a completely separate connection to prove it was actually,
durably persisted -- not just that no exception was raised.

Usage
-----
    python3 -m pytest labs/tests/test_wasm_persistence.py -v

Requires Playwright with Chromium installed:
    pip install playwright
    python3 -m playwright install chromium
"""

from __future__ import annotations

import functools
import http.server
import json
import shutil
import socketserver
import threading
from pathlib import Path

import pytest


STATE_PY = (
    Path(__file__).resolve().parents[2] / "mlsysim" / "mlsysim" / "labs" / "state.py"
)
TRIALS = 5
PYODIDE_CDN = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/pyodide.js"

PAGE_HTML = f"""<!doctype html>
<html><head><meta charset="utf-8"></head>
<body>
<script src="{PYODIDE_CDN}"></script>
<script type="module">
  async function main() {{
    try {{
      const pyodide = await loadPyodide();
      window.__pyodide = pyodide;
      const stateSrc = await (await fetch("state.py")).text();
      pyodide.globals.set("__state_src", stateSrc);
      await pyodide.runPythonAsync(`
import sys, types
_mod = types.ModuleType("state_under_test")
sys.modules[_mod.__name__] = _mod
exec(__state_src, _mod.__dict__)
DesignLedger = _mod.DesignLedger
`);
      window.__ready = true;
    }} catch (e) {{
      window.__initError = String(e && e.stack ? e.stack : e);
    }}
  }}
  main();
</script>
</body></html>
"""


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return


def _start_server(directory: Path):
    handler = functools.partial(_QuietHandler, directory=str(directory))
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@pytest.fixture(scope="module")
def served_dir(tmp_path_factory):
    if not STATE_PY.is_file():
        pytest.skip(f"state.py not found at {STATE_PY}")

    directory = tmp_path_factory.mktemp("wasm-persistence")
    shutil.copy(STATE_PY, directory / "state.py")
    (directory / "index.html").write_text(PAGE_HTML, encoding="utf-8")

    server = _start_server(directory)
    try:
        yield directory, server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def test_design_ledger_save_async_persists_in_real_indexeddb(served_dir):
    """save_async() must actually persist to IndexedDB when called as a
    bound DesignLedger method -- not just report success.

    Runs TRIALS independent attempts, each against a freshly-cleared
    IndexedDB, reading back through a *separate* connection each time to
    verify durability rather than trusting save_async()'s return value.
    """
    from playwright.sync_api import sync_playwright

    failures: list[int] = []
    _, port = served_dir

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        try:
            for i in range(TRIALS):
                page = context.new_page()
                init_errors: list[str] = []
                page.on(
                    "pageerror",
                    lambda exc, errors=init_errors: errors.append(str(exc)),
                )

                page.goto(f"http://127.0.0.1:{port}/index.html")
                page.wait_for_function(
                    "window.__ready === true || window.__initError",
                    timeout=30_000,
                )
                init_error = page.evaluate("window.__initError || null")
                assert not init_error, f"Pyodide init failed: {init_error}"
                assert not init_errors, (
                    f"Uncaught page errors during init: {init_errors}"
                )

                # Clear any prior IndexedDB state for a clean trial.
                page.evaluate(
                    """
                    () => new Promise((resolve) => {
                        const req = indexedDB.deleteDatabase("mlsys_ledger_db");
                        req.onsuccess = req.onerror = req.onblocked = () => resolve();
                    })
                    """
                )

                page.evaluate(
                    f"""
                    async () => {{
                        const pyodide = window.__pyodide;
                        await pyodide.runPythonAsync(`
ledger = DesignLedger()
ledger._state.track = "trial-{i}"
ledger._state.current_step = 1
ledger._state.history[1] = {{"trial": {i}}}
await ledger.save_async()
`);
                    }}
                    """
                )

                persisted = page.evaluate(
                    """
                    () => new Promise((resolve, reject) => {
                        const req = indexedDB.open("mlsys_ledger_db", 1);
                        req.onsuccess = (e) => {
                            const db = e.target.result;
                            const tx = db.transaction("ledger", "readonly");
                            const getReq = tx.objectStore("ledger").get("mlsys_design_ledger");
                            getReq.onsuccess = () => {
                                db.close();
                                resolve(getReq.result !== undefined && getReq.result !== null);
                            };
                            getReq.onerror = () => { db.close(); reject(getReq.error); };
                        };
                        req.onerror = () => reject(req.error);
                    })
                    """
                )

                if not persisted:
                    failures.append(i)
                page.close()
        finally:
            context.close()
            browser.close()

    assert not failures, (
        f"DesignLedger.save_async() failed to durably persist to IndexedDB "
        f"on trial(s) {failures} of {TRIALS} -- it reported success but the "
        f"write was lost. This is the exact silent-data-loss failure mode "
        f"from #1985 / PR #1988. A mocked test cannot catch this; only a "
        f"real Pyodide + IndexedDB check like this one can."
    )


def test_load_async_corrupt_record_sets_last_load_error(served_dir):
    """A stored record exists but is corrupt JSON -- json.loads() raising
    is a Python-side failure independent of the JS resolve/reject shape,
    so last_load_error must be populated regardless of #1988's status."""
    from playwright.sync_api import sync_playwright

    _, port = served_dir

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        try:
            page = context.new_page()
            init_errors: list[str] = []
            page.on(
                "pageerror",
                lambda exc, errors=init_errors: errors.append(str(exc)),
            )

            page.goto(f"http://127.0.0.1:{port}/index.html")
            page.wait_for_function(
                "window.__ready === true || window.__initError", timeout=30_000
            )
            init_error = page.evaluate("window.__initError || null")
            assert not init_error, f"Pyodide init failed: {init_error}"
            assert not init_errors, f"Uncaught page errors during init: {init_errors}"

            page.evaluate(
                """
                () => new Promise((resolve) => {
                    const req = indexedDB.deleteDatabase("mlsys_ledger_db");
                    req.onsuccess = req.onerror = req.onblocked = () => resolve();
                })
                """
            )

            # Seed a corrupt record directly at the storage layer.
            page.evaluate(
                """
                () => new Promise((resolve, reject) => {
                    const req = indexedDB.open("mlsys_ledger_db", 1);
                    req.onupgradeneeded = (e) => {
                        const db = e.target.result;
                        if (!db.objectStoreNames.contains("ledger")) {
                            db.createObjectStore("ledger");
                        }
                    };
                    req.onsuccess = (e) => {
                        const db = e.target.result;
                        const tx = db.transaction("ledger", "readwrite");
                        tx.objectStore("ledger").put("{not valid json", "mlsys_design_ledger");
                        tx.oncomplete = () => { db.close(); resolve(); };
                        tx.onerror = () => { db.close(); reject(tx.error); };
                    };
                    req.onerror = () => reject(req.error);
                })
                """
            )

            result = page.evaluate(
                """
                async () => {
                    const pyodide = window.__pyodide;
                    return await pyodide.runPythonAsync(`
import json
ledger = DesignLedger()
await ledger.load_async()
json.dumps({"error": ledger.last_load_error})
`);
                }
                """
            )
            page.close()
        finally:
            context.close()
            browser.close()

    parsed = json.loads(result)
    assert parsed["error"] is not None, (
        "load_async() must surface a corrupt-JSON read failure via "
        "last_load_error instead of silently returning a blank LedgerState()."
    )


def test_load_async_synchronous_indexeddb_open_throw_sets_last_load_error(served_dir):
    """indexedDB.open() throwing synchronously must reject the Promise
    (per the Promise constructor spec) and propagate to last_load_error --
    true today even against the pre-#1988 resolve(null)-style onerror
    handlers, since this never reaches those handlers at all."""
    from playwright.sync_api import sync_playwright

    _, port = served_dir

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        try:
            page = context.new_page()
            init_errors: list[str] = []
            page.on(
                "pageerror",
                lambda exc, errors=init_errors: errors.append(str(exc)),
            )

            page.goto(f"http://127.0.0.1:{port}/index.html")
            page.wait_for_function(
                "window.__ready === true || window.__initError", timeout=30_000
            )
            init_error = page.evaluate("window.__initError || null")
            assert not init_error, f"Pyodide init failed: {init_error}"
            assert not init_errors, f"Uncaught page errors during init: {init_errors}"

            page.evaluate(
                """() => {
                    window.indexedDB.open = () => {
                        throw new Error('Simulated synchronous IndexedDB failure');
                    };
                }"""
            )

            result = page.evaluate(
                """
                async () => {
                    const pyodide = window.__pyodide;
                    return await pyodide.runPythonAsync(`
import json
ledger = DesignLedger()
await ledger.load_async()
json.dumps({"error": ledger.last_load_error})
`);
                }
                """
            )
            page.close()
        finally:
            context.close()
            browser.close()

    parsed = json.loads(result)
    assert parsed["error"] is not None
    assert "Simulated synchronous IndexedDB failure" in parsed["error"]