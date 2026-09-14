#!/usr/bin/env python3
"""Autonomous Swarm Conductor for MLSysBook Co-Labs.

Orchestrates the build, simulation validation, static AST checks, headless
marimo compilation, and Playwright visual snapshot verification for labs
across Volumes I-IV.
"""

from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LABS_ROOT = REPO_ROOT / "labs"


@dataclass
class SwarmAuditResult:
    lab_path: str
    static_tests_passed: bool
    html_exported: bool
    visual_screenshot_path: str | None
    error_message: str | None = None


def run_command(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def verify_lab(lab_path: Path, output_dir: Path) -> SwarmAuditResult:
    rel_path = str(lab_path.relative_to(LABS_ROOT))
    print(f"\n[Conductor] ── Auditing {rel_path} ──")

    # Step 1: Static AST & Reactive checks
    print("  [1/3] Running static AST and Marimo dataflow tests...")
    pytest_bin = LABS_ROOT / ".venv" / "bin" / "pytest"
    code, out = run_command([
        str(pytest_bin),
        str(LABS_ROOT / "tests" / "test_static.py"),
        "-k", lab_path.stem,
        "-v",
    ], cwd=LABS_ROOT)

    if code != 0:
        print(f"  ❌ Static tests failed:\n{out}")
        return SwarmAuditResult(
            lab_path=rel_path,
            static_tests_passed=False,
            html_exported=False,
            visual_screenshot_path=None,
            error_message="Static tests failed",
        )
    print("  ✅ Static tests passed.")

    # Step 2: Headless HTML export
    print("  [2/3] Compiling headless Marimo HTML...")
    marimo_bin = LABS_ROOT / ".venv" / "bin" / "marimo"
    output_html = output_dir / f"{lab_path.stem}.html"
    code, out = run_command([
        str(marimo_bin),
        "export",
        "html",
        str(lab_path),
        "--no-include-code",
        "-o",
        str(output_html),
    ], cwd=LABS_ROOT)

    if code != 0:
        print(f"  ❌ Marimo export failed:\n{out}")
        return SwarmAuditResult(
            lab_path=rel_path,
            static_tests_passed=True,
            html_exported=False,
            visual_screenshot_path=None,
            error_message="Marimo HTML export failed",
        )
    print(f"  ✅ Compiled to {output_html}")

    # Step 3: Playwright visual snapshot
    print("  [3/3] Capturing Playwright visual snapshot...")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  ⚠️ Playwright not installed. Skipping visual screenshot.")
        return SwarmAuditResult(
            lab_path=rel_path,
            static_tests_passed=True,
            html_exported=True,
            visual_screenshot_path=None,
            error_message="Playwright unavailable",
        )

    import functools

    screenshot_path = output_dir / f"{lab_path.stem}_visual.png"
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(output_dir))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]

    srv = threading.Thread(target=httpd.serve_forever)
    srv.daemon = True
    srv.start()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1200, "height": 1800})
            url = f"http://127.0.0.1:{port}/{output_html.name}"
            page.goto(url, wait_until="networkidle", timeout=25000)
            page.wait_for_timeout(3000)
            page.screenshot(path=str(screenshot_path), full_page=True)
            browser.close()
        print(f"  ✅ Visual snapshot saved to {screenshot_path}")
    except Exception as exc:
        print(f"  ⚠️ Visual capture warning: {exc}")
        screenshot_path = None
    finally:
        httpd.shutdown()
        httpd.server_close()

    return SwarmAuditResult(
        lab_path=rel_path,
        static_tests_passed=True,
        html_exported=True,
        visual_screenshot_path=str(screenshot_path) if screenshot_path else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", type=str, help="Specific lab path to audit (e.g. vol1/lab_02_ml_systems.py)")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/swarm_audits"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.lab:
        labs = [LABS_ROOT / args.lab]
    else:
        labs = sorted(LABS_ROOT.glob("vol*/lab_*.py"))

    results = []
    for lab in labs:
        res = verify_lab(lab, args.output_dir)
        results.append(res)

    print("\n" + "=" * 50)
    print("SWARM CONDUCTION SUMMARY")
    print("=" * 50)
    all_passed = True
    for r in results:
        status = "PASSED" if (r.static_tests_passed and r.html_exported) else "FAILED"
        if status == "FAILED":
            all_passed = False
        print(f"{r.lab_path:35} [{status}] (Screenshot: {r.visual_screenshot_path})")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
