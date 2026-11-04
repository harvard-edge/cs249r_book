"""Exercise taxonomy navigation, search, clipboard, and diagram accessibility.

Build and serve site/_build first. Requires Playwright and Chromium.
Usage: python3 site/scripts/check_taxonomy_ui.py http://localhost:8768/taxonomy/
"""

import sys
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import expect, sync_playwright


def check(url: str) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for width, dark in ((1440, False), (390, False), (1440, True)):
            page = browser.new_page(
                viewport={"width": width, "height": 1000}, reduced_motion="reduce"
            )
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.context.grant_permissions(["clipboard-read", "clipboard-write"])
            for fragment in (
                "", "view=matrix", "area=serving", "subject=structured-generation",
                "concept=mlsys.kv-cache", "concept=mlsys.functional-insufficiency",
                "view=glossary", "view=guides",
            ):
                page.goto(f"{url}#{fragment}", wait_until="networkidle")
                if dark:
                    page.evaluate("""() => {
                        if (!document.body.classList.contains('quarto-dark'))
                            window.quartoToggleColorScheme();
                    }""")
                assert not page.evaluate(
                    "document.documentElement.scrollWidth > innerWidth"
                ), (width, dark, fragment, "horizontal overflow")
                if fragment == "area=serving":
                    expect(page.locator("#bok-entry-body h2")).to_contain_text("Deployment")
                if fragment.startswith("subject="):
                    assert page.locator(".bok-tree-box").evaluate("e => e.open") == (width > 900)

            search = page.locator("#bok-search")
            search.fill("KV cache")
            search.press("ArrowDown")
            active = search.get_attribute("aria-activedescendant")
            assert active
            expect(page.locator(f"#{active} .bok-res-description")).not_to_be_empty()
            search.press("Enter")
            expect(page.locator("#bok-entry-body h2")).to_have_text("KV cache")
            for kind in ("id", "link"):
                page.locator(f'[data-copy-concept="{kind}"]').click()
                copied = page.evaluate("navigator.clipboard.readText()")
                assert copied == "mlsys.kv-cache" if kind == "id" else copied.endswith("#concept=mlsys.kv-cache")
            feedback = page.locator(".bok-entry-actions a").get_attribute("href")
            assert "mlsys.kv-cache" in parse_qs(urlparse(feedback).query)["target"][0]

            page.goto(url, wait_until="networkidle")
            page.locator('#bok-explore-area').select_option('serving')
            expect(page.locator('#bok-coupling-panel')).to_contain_text('Deployment')
            page.locator('[data-coupling-mode="dsm"]').click()
            page.locator('#bok-explore-target').select_option('system-models')
            expect(page.locator('#bok-coupling-panel')).to_contain_text('System models')
            page.locator('[data-grid-row="0"][data-grid-col="0"]').focus()
            page.keyboard.press("ArrowRight")
            page.keyboard.press("Enter")
            assert page.evaluate("document.activeElement.dataset.gridCol") == "1"
            expect(page.locator('[data-grid-row="0"][data-grid-col="1"]')).to_have_class("viz-dsm-cell is-selected")
            figure = page.locator("#bok-coupling-figure")
            page.locator('[data-viz-expand="bok-coupling-figure"]').click()
            expect(figure).to_have_attribute("aria-modal", "true")
            page.keyboard.press("Tab")
            assert page.evaluate('!!document.activeElement.closest("#bok-coupling-figure")')
            page.keyboard.press("Escape")
            expect(figure).not_to_have_attribute("aria-modal", "true")
            assert not errors, errors
            page.close()
        browser.close()
    print("Taxonomy UI checks passed: desktop, mobile, and dark mode.")


if __name__ == "__main__":
    check(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8768/taxonomy/")
