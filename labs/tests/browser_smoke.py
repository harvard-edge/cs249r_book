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
import socketserver
import sys
import threading
import time
from pathlib import Path


BOOT_TIMEOUT_MS = 180_000  # 3 min for Pyodide + wheel install + cell exec
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

MARIMO_READY_SELECTORS = [
    'marimo-island',         # marimo root custom element
    '[role="tab"]',          # mo.ui.tabs — every lab uses this
    '.marimo-cell',           # generic cell wrapper fallback
]


def verify_lab(page, name: str, url: str) -> list[str]:
    """Navigate to a lab and wait for a marimo DOM signal. Return error list."""
    errors: list[str] = []

    def record_error(exc):
        errors.append(f"[pageerror] {exc}")

    def record_console(msg):
        if msg.type == "error":
            errors.append(f"[console.error] {msg.text}")

    page.on("pageerror", record_error)
    page.on("console", record_console)

    print(f"  → {name}: navigating to {url}", flush=True)
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)

    start = time.monotonic()
    selector = ", ".join(MARIMO_READY_SELECTORS)
    try:
        page.wait_for_selector(selector, timeout=BOOT_TIMEOUT_MS, state="attached")
    except Exception as exc:  # noqa: BLE001 — Playwright raises a generic error
        elapsed = time.monotonic() - start
        errors.append(
            f"[timeout] no marimo DOM after {elapsed:.0f}s "
            f"(waited for: {selector}): {exc}"
        )
        return errors

    elapsed = time.monotonic() - start
    print(f"  ✅ {name}: marimo DOM rendered in {elapsed:.1f}s", flush=True)
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
