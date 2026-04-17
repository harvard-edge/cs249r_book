"""
Level 5: Browser-level WASM Smoke Test
=======================================

Launches a real headless Chromium via Playwright against WASM-exported labs
served behind the cross-origin isolation headers Pyodide + SharedArrayBuffer
require. Validates that:

  - Pyodide actually initializes in a real browser (not just Node)
  - marimo renders interactive DOM (tab/cell elements)
  - No uncaught page errors are raised during boot

Why this exists
---------------
Static tests, engine tests, and Node-Pyodide wheel tests can all pass while a
lab is broken in the browser. lab_05 shipped with plotly imported before
`micropip.install(...)` — caught only by a real browser (#1353). This is the
regression guard so that never happens silently again.

Usage
-----
    python3 labs/tests/browser_smoke.py --labs-dir /tmp/wasm-smoke

Expects the `--labs-dir` directory to contain subdirectories, one per lab,
each with an `index.html` produced by `marimo export html-wasm`.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import re
import socketserver
import sys
import threading
import time
from pathlib import Path


BOOT_TIMEOUT_MS = 180_000  # 3 min for Pyodide + wheel install + cell exec
SHELL_TIMEOUT_MS = 30_000  # static shell should paint almost immediately
POST_IDLE_SETTLE_S = 5.0   # grace for synchronous cell output after network idle
PORT = 8765


# ── COOP/COEP HTTP server ───────────────────────────────────────────────────
#
# SharedArrayBuffer (required by Pyodide for threading) is only enabled for
# documents served with cross-origin isolation headers. Pyodide boots without
# SAB, but threaded workloads and some wheels break. Matching the production
# dev-preview headers here keeps parity.

class CrossOriginIsolatedHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()

    def log_message(self, fmt, *args):  # noqa: A003 — stdlib override
        # Keep CI logs quiet; errors still go to stderr via log_error.
        return


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_server(root: Path, port: int) -> ThreadedServer:
    handler = functools.partial(CrossOriginIsolatedHandler, directory=str(root))
    server = ThreadedServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# ── Browser driver ──────────────────────────────────────────────────────────

# Marimo WASM export serializes pre-run cell outputs into the HTML shell, so
# these selectors attach to the DOM almost immediately — long before Pyodide
# has actually executed anything. Matching them only proves the page loaded,
# NOT that Python ran. We use them as the fast-path shell check, then fall
# through to a network-idle wait that does not return until Pyodide has
# downloaded its runtime and every wheel the lab needs.
SHELL_SELECTORS = [
    'marimo-island',
    '[role="tab"]',
    '.marimo-cell',
]


# Marimo routes Python cell stderr to console.log, not console.error. The
# JSON blob it emits after a traceback is the cleanest machine-readable
# signal: a line containing `{"type":"exception", ...}` means a cell raised
# uncaught. This matches the shape marimo 0.23.x emits; kept permissive so
# minor schema drift does not silently mask errors.
PYTHON_EXCEPTION_RE = re.compile(r'\{"type"\s*:\s*"exception"[^}]*\}')


def _extract_python_exception(text: str) -> str | None:
    """If this console line carries a marimo-structured Python exception,
    return a compact one-line summary; otherwise None."""
    match = PYTHON_EXCEPTION_RE.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return match.group(0)[:200]
    exc_type = payload.get("exception_type") or "Exception"
    msg = payload.get("msg") or ""
    return f"{exc_type}: {msg}"


def verify_lab(page, name: str, url: str) -> list[str]:
    """Navigate to a lab, let Pyodide boot, then report any captured errors."""
    errors: list[str] = []

    def record_error(exc):
        errors.append(f"[pageerror] {exc}")

    def record_console(msg):
        # Marimo emits Python exceptions through styled console.log, not
        # console.error, so we have to scan every log line for the structured
        # exception marker instead of filtering on msg.type.
        py_err = _extract_python_exception(msg.text)
        if py_err:
            errors.append(f"[python] {py_err}")
        elif msg.type == "error":
            errors.append(f"[console.error] {msg.text[:300]}")

    page.on("pageerror", record_error)
    page.on("console", record_console)

    print(f"  → {name}: navigating to {url}", flush=True)
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)

    # Phase 1: static shell must paint quickly. If this fails the export is
    # broken — no point waiting the full Pyodide budget.
    shell_selector = ", ".join(SHELL_SELECTORS)
    try:
        page.wait_for_selector(
            shell_selector, timeout=SHELL_TIMEOUT_MS, state="attached"
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(
            f"[shell-timeout] marimo shell never attached "
            f"(waited for: {shell_selector}): {exc}"
        )
        return errors
    print(f"  …  {name}: shell rendered, waiting for Pyodide to settle", flush=True)

    # Phase 2: Pyodide downloads runtime + wheels asynchronously. Network idle
    # only fires once every fetch has resolved, which in practice means the
    # full micropip.install(...) chain has completed. Without this wait the
    # #1353-class bug (plotly imported before micropip.install) would go
    # undetected — marimo's shell renders before Python executes.
    pyodide_start = time.monotonic()
    try:
        page.wait_for_load_state("networkidle", timeout=BOOT_TIMEOUT_MS)
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - pyodide_start
        errors.append(
            f"[pyodide-timeout] network never went idle after {elapsed:.0f}s "
            f"(Pyodide likely did not boot): {exc}"
        )
        return errors

    # Phase 3: small settle buffer so synchronous post-load cell work (e.g.
    # plotly figure construction after micropip.install completes) has time
    # to emit its errors to the console before we tally.
    page.wait_for_timeout(int(POST_IDLE_SETTLE_S * 1000))

    elapsed = time.monotonic() - pyodide_start
    print(
        f"  ✅ {name}: Pyodide settled in {elapsed:.1f}s "
        f"(captured errors: {len(errors)})",
        flush=True,
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labs-dir",
        required=True,
        help="Directory containing exported lab subdirs, each with index.html.",
    )
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    labs_dir = Path(args.labs_dir).resolve()
    if not labs_dir.is_dir():
        print(f"❌ labs-dir does not exist: {labs_dir}", file=sys.stderr)
        return 2

    lab_names = sorted(
        p.name for p in labs_dir.iterdir() if (p / "index.html").is_file()
    )
    if not lab_names:
        print(f"❌ no labs with index.html in {labs_dir}", file=sys.stderr)
        return 2

    # Playwright is imported here so the module loads without it for --help
    from playwright.sync_api import sync_playwright

    print(f"🌐 serving {labs_dir} with COEP/COOP on :{args.port}", flush=True)
    server = start_server(labs_dir, args.port)
    all_errors: dict[str, list[str]] = {}

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=["--enable-features=SharedArrayBuffer"],
            )
            context = browser.new_context()
            for name in lab_names:
                page = context.new_page()
                url = f"http://127.0.0.1:{args.port}/{name}/index.html"
                errors = verify_lab(page, name, url)
                if errors:
                    all_errors[name] = errors
                page.close()
            context.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()

    if all_errors:
        print("", flush=True)
        print("❌ browser smoke failed:", flush=True)
        for name, errs in all_errors.items():
            print(f"\n  {name}:", flush=True)
            for err in errs:
                print(f"    - {err}", flush=True)
        return 1

    print(f"\n✅ all {len(lab_names)} labs booted in browser cleanly", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
