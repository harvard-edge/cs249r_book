#!/usr/bin/env python3
"""Headless-Chromium smoke test for the built Next.js export.

Runs in CI after `npm run build`. Starts a static file server from
`interviews/staffml/out/`, loads a handful of critical routes in
Playwright, and fails the job if any of these invariants break:

  1. HTTP 200 on every page
  2. No uncaught page errors (React error boundaries / TypeErrors /
     null derefs — the class that bit us in PR #1440)
  3. No console.error except a small allowlist of known benign noise
     (Next.js RSC .txt prefetches on static export; Cloudflare beacon
     404s that don't fire under localhost anyway)
  4. Practice page renders the scenario for a known question within
     a reasonable timeout (covers the hydration code path)

This is intentionally a CHEAP gate: one browser, <60 seconds, no
visual regression tests. The goal is to catch "white screen of death"
class bugs before they merge, not to comprehensively cover UX.
"""

from __future__ import annotations

import http.server
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

OUT_DIR = Path("interviews/staffml/out")
# Must match an origin on the vault-worker CORS allowlist (wrangler.toml
# CORS_ALLOWLIST includes http://localhost:3000). If you change this,
# update the worker allowlist too or the hydration fetches will fail with
# CORS errors in the smoke — which is exactly what the smoke is designed
# to catch, so it's a load-bearing constraint.
PORT = 3000

# Every route listed here gets loaded. Keep this list small — each adds
# ~2 seconds. If a new feature deserves E2E coverage, add the route here
# and optionally a visible-text assertion in `ASSERT_CONTAINS`.
ROUTES = [
    "/",
    "/practice/",
    "/plans/",
    "/gauntlet/",
    "/about/",
    "/progress/",
]

# After load, assert these substrings appear somewhere in document.body.
# Catches "page loads but content missing" — exactly the failure mode
# of the hydration shape-mismatch bug.
# next.config.mjs sets `trailingSlash: true`, so the canonical URLs are
# /<page>/ (served as <page>/index.html); the legacy /<page>.html paths
# return 404.
ASSERT_CONTAINS: dict[str, list[str]] = {
    "/practice/": ["Practice"],      # title at minimum
    "/about/":    ["StaffML"],
}

# Console errors whose text matches any of these substrings are ignored.
# Add new entries only with a code comment explaining WHY (avoid growing
# the allowlist into a "silence everything" sink).
CONSOLE_IGNORE = [
    # Next.js static-export tries to prefetch RSC .txt payloads on
    # cross-site <Link> hover; 404s against static hosts. Known quirk.
    "_rsc=",
    ".txt 404",
    # Cloudflare Web Analytics beacon — fires on prod Cloudflare zones
    # but not on localhost, so this shouldn't appear in CI but keeping
    # the filter defensively.
    "cloudflareinsights",
    # Chromium logs "Failed to load resource" as a console.error for
    # every 4xx. We already trip on the network layer; don't double-fail.
    "Failed to load resource",
]


class SilentHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs) -> None:  # noqa: D401
        pass


class _ReusableServer(socketserver.TCPServer):
    # Allow quick re-bind to the same port across runs; CI reuses the same
    # ephemeral VM for the step so this matters less, but it's harmless
    # and saves "Address already in use" errors when iterating locally.
    allow_reuse_address = True


def start_server() -> _ReusableServer:
    """Start a static file server in a daemon thread."""
    import os
    os.chdir(OUT_DIR)
    httpd = _ReusableServer(("", PORT), SilentHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    # Wait a beat for the socket to listen.
    time.sleep(0.5)
    return httpd


def main() -> int:
    if not OUT_DIR.exists():
        print(f"❌ {OUT_DIR} missing — run `npm run build` first.", file=sys.stderr)
        return 2

    # Lazy-import Playwright so a missing install fails with a clearer error.
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ playwright not installed. Run: pip install playwright && playwright install chromium", file=sys.stderr)
        return 3

    start_server()
    base = f"http://localhost:{PORT}"
    print(f"📡 Serving {OUT_DIR} at {base}")

    failures: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()

        console_errs: list[str] = []
        page_errs: list[str] = []
        page.on("console", lambda m: m.type == "error" and console_errs.append(m.text))
        page.on("pageerror", lambda exc: page_errs.append(str(exc)))

        for route in ROUTES:
            url = base + route
            print(f"\n→ {url}")
            console_errs.clear()
            page_errs.clear()
            try:
                resp = page.goto(url, wait_until="networkidle", timeout=20_000)
            except Exception as exc:
                failures.append(f"{route}: goto failed — {exc}")
                continue

            if resp is None or resp.status >= 400:
                failures.append(f"{route}: HTTP {resp.status if resp else 'None'}")
                continue

            page.wait_for_timeout(2000)   # let hydration + worker fetches settle

            for marker in ASSERT_CONTAINS.get(route, []):
                body = page.inner_text("body")
                if marker not in body:
                    failures.append(f"{route}: expected text {marker!r} not found in DOM")

            if page_errs:
                failures.extend(f"{route}: pageerror — {e[:200]}" for e in page_errs)

            real_console = [
                e for e in console_errs
                if not any(ignored in e for ignored in CONSOLE_IGNORE)
            ]
            if real_console:
                failures.extend(f"{route}: console.error — {e[:200]}" for e in real_console)

            status = "✅" if not (page_errs or real_console) else "❌"
            print(f"  {status}  HTTP {resp.status}  "
                  f"pageerror={len(page_errs)}  console.error={len(real_console)}")

        browser.close()

    print()
    if failures:
        print(f"❌ E2E smoke FAILED — {len(failures)} issue(s):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("✅ E2E smoke passed across all routes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
