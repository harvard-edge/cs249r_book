"""Capture Playwright screenshots of the labs landing page for review."""

import http.server
import os
import socketserver
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

LABS_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = LABS_ROOT / "_build"
SHOT_DIR = LABS_ROOT / "_screenshots"
PORT = 8765

SHOT_DIR.mkdir(exist_ok=True)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs):
        return


def serve_build_dir():
    os.chdir(BUILD_DIR)
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("", PORT), QuietHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def capture(page, locator_str: str, fname: str, scroll: bool = True):
    loc = page.locator(locator_str).first
    if scroll:
        loc.scroll_into_view_if_needed()
        page.wait_for_timeout(200)
    loc.screenshot(path=str(SHOT_DIR / fname))


def main():
    httpd = serve_build_dir()
    try:
        url = f"http://localhost:{PORT}/index.html"
        with sync_playwright() as p:
            browser = p.chromium.launch()

            # Desktop full page + element shots
            ctx = browser.new_context(viewport={"width": 1440, "height": 900})
            page = ctx.new_page()
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(400)
            page.screenshot(
                path=str(SHOT_DIR / "01_desktop_full.png"),
                full_page=True,
            )
            capture(page, ".lab-window", "02_desktop_hero.png")
            capture(page, ".lab-cluster[data-accent='blue']", "03_desktop_vol1_cluster1.png")
            capture(page, ".lab-cluster[data-accent='amber']", "04_desktop_vol1_cluster2.png")
            capture(page, ".lab-card.capstone", "05_desktop_capstone.png")
            capture(page, ".inventory-legend", "09_desktop_legend.png")
            # The whole Volume I inventory section
            capture(
                page,
                "h3:has-text('Volume I')",
                "06_desktop_vol1_heading.png",
                scroll=False,
            )
            ctx.close()

            # Tablet
            ctx = browser.new_context(viewport={"width": 900, "height": 1200})
            page = ctx.new_page()
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(400)
            page.screenshot(
                path=str(SHOT_DIR / "07_tablet_full.png"),
                full_page=True,
            )
            ctx.close()

            # Mobile
            ctx = browser.new_context(viewport={"width": 400, "height": 800})
            page = ctx.new_page()
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(400)
            page.screenshot(
                path=str(SHOT_DIR / "08_mobile_full.png"),
                full_page=True,
            )
            ctx.close()

            browser.close()

        print(f"Screenshots written to {SHOT_DIR}")
        for p in sorted(SHOT_DIR.glob("*.png")):
            print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()
