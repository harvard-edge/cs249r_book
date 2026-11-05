#!/usr/bin/env python3
"""Exercise Lab 02 discussion state in a running native or WASM notebook.

Run with python3 labs/tools/check_placement_discussion.py --url URL.
2026-10-20: a real browser exposed Marimo state self-loop suppression that
headless app.run() tests missed: attaching an interpretation never updated UI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from playwright.sync_api import expect, sync_playwright


def check(url: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
        page.set_default_timeout(30_000)
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.get_by_role("tab", name="Part A · Locality", exact=True).wait_for(timeout=180_000)
        page.get_by_role("radio", name="Connection upgrade", exact=True).wait_for(timeout=180_000)
        # Each run uses the same supplied TinyML study, even if a prior journey was saved.
        page.locator("select").first.select_option(label="TinyML")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(output / "opening-desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(600)
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth + 1")
        page.screenshot(path=str(output / "opening-mobile.png"), full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.wait_for_timeout(600)

        def button(label: str) -> None:
            page.get_by_role("button", name=label, exact=True).last.click()
            page.wait_for_timeout(1000)

        def tab(label: str) -> None:
            page.get_by_role("tab", name=label, exact=True).last.click()
            page.wait_for_timeout(600)

        def textareas():
            return page.locator("textarea:not([readonly])").filter(visible=True)

        def attach(part: str, explanation: str) -> None:
            area = textareas().first
            area.fill(explanation)
            area.press("Tab")
            page.wait_for_timeout(600)
            button(f"Attach {part} interpretation")
            page.get_by_text("Interpretation attached", exact=True).last.wait_for()

        page.get_by_role("radio", name="Connection upgrade", exact=True).check()
        button("Lock A prediction")
        button("Pin this network run")
        page.locator("select").nth(1).select_option(index=0)
        page.wait_for_timeout(600)
        button("Save locality contrast")
        attach("A", "Payload changed at fixed link. Upload shrank; propagation remains. Target tail latency is unmeasured.")
        # Editing must invalidate the association; a fresh explicit attachment must work.
        area = textareas().first
        area.fill("Edited explanation: reducing payload changes transfer, not propagation. Measure the target path before approval.")
        area.press("Tab")
        page.get_by_text("Next · attach your interpretation", exact=True).last.wait_for()
        button("Attach A interpretation")
        page.get_by_text("Interpretation attached", exact=True).last.wait_for()
        # Recapturing must preserve history and invalidate the old explanation.
        button("Pin this network run")
        page.locator("select").nth(1).select_option(index=2)
        page.wait_for_timeout(600)
        button("Save locality contrast")
        page.get_by_text("Next · attach your interpretation", exact=True).last.wait_for()
        attach("A", "The second saved pair increases payload at fixed link. Transfer worsens; the first failed investigation remains evidence.")

        tab("Part B · Resources")
        page.get_by_role("radio", name="Arithmetic capability", exact=True).check()
        button("Lock B prediction")
        button("Pin this resource attempt")
        page.get_by_role("combobox").last.select_option(label="More working memory")
        page.wait_for_timeout(600)
        button("Save resource contrast")
        attach("B", "Working memory repairs capacity for the larger workload; faster arithmetic did not. Check flash independently.")

        tab("Part C · Operation")
        page.get_by_role("radio", name="Reduce observation", exact=True).check()
        button("Lock C prediction")
        button("Pin this operating policy")
        page.get_by_role("slider").last.focus()
        page.get_by_role("slider").last.press("Home")
        page.wait_for_timeout(600)
        button("Save operation contrast")
        attach("C", "Lower activity saves energy and power but always off fails observation. More reserve cannot repair service.")

        tab("Part D · Placement")
        page.get_by_role("radio", name="Raw remote", exact=True).check()
        button("Lock D prediction")
        page.get_by_role("checkbox", name="Raw input may leave the endpoint", exact=True).uncheck()
        page.wait_for_timeout(600)
        button("Save stressed placement decision")
        attach("D", "Withdrawal of permission rejects raw remote. Hold because saved service already fails; restoring permission alone is insufficient.")
        tab("Synthesis")
        page.get_by_role("heading", name="Team decision so far", exact=True).last.wait_for()
        page.get_by_text("Hold for further evidence", exact=True).last.wait_for()
        for label in ("Carry forward", "Still open", "Next"):
            page.locator(".nb-bridge").last.get_by_role("heading", name=label, exact=True).wait_for()
        page.get_by_role("heading", name="Team decision so far", exact=True).last.scroll_into_view_if_needed()
        page.screenshot(path=str(output / "team-decision-desktop.png"), full_page=True)
        textareas().nth(0).fill("Hold: service fails despite improved energy. Capacity repair does not replace observation; raw remote also fails permission.")
        area = textareas().nth(1)
        area.fill("Measure target sensing, radio, memory and tail latency. Rerun after the observation policy changes; lifecycle costs are incomplete.")
        area.press("Tab")
        page.get_by_role("heading", name="Your discussion", exact=True).last.wait_for()
        with page.expect_download() as download:
            page.get_by_role("link", name="Download JSON snapshot", exact=True).last.click()
        report_path = output / "discussion.json"
        download.value.save_as(str(report_path))
        report = json.loads(report_path.read_text())
        snapshot = report["result_snapshot"]
        assert snapshot["decision"]["chosen"] == "hold"
        brief = snapshot["team_decision_brief"]
        assert brief["state"] == "hold"
        assert brief["decision"] == "Hold for further evidence"
        assert brief["project"] and brief["target_evidence_needed"] and brief["next_chapter_question"]
        assert "unresolved" in brief["scope"]
        assert snapshot["journey"]["scenario_id"] == "wearable-ring.v1"
        assert snapshot["journey"]["chapter"] == 2
        history = snapshot["investigation_history"]["A"]
        assert len(history) == 2
        assert history[0]["prediction"] == history[1]["prediction"] == "Connection upgrade"
        assert history[0]["result"]["inputs"]["payload_mb"] != history[1]["result"]["inputs"]["payload_mb"]
        assert set(snapshot["interpretations"]) == set("ABCD")
        button("Save discussion")
        expect(page.locator(".nb-status").filter(has_text="Download remains available.").last.locator("strong")).to_have_text("Saved")
        area.fill("Revised uncertainty: measure target sensing and radio power, then repair the observation policy before reconsidering placement.")
        area.press("Tab")
        expect(page.locator(".nb-status").filter(has_text="Download remains available.").last.locator("strong")).to_have_text("Ready to save")
        page.screenshot(path=str(output / "discussion-desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(600)
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth + 1")
        page.screenshot(path=str(output / "discussion-mobile.png"), full_page=True)
        assert not page.locator("marimo-error-output").count()
        assert not errors, errors
        browser.close()
    print(f"Placement discussion browser checks passed. Artifacts: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Running Lab 02 native or WASM URL")
    parser.add_argument("--output-dir", type=Path, help="Directory for the report and screenshots")
    arguments = parser.parse_args()
    output = arguments.output_dir or Path(tempfile.mkdtemp(prefix="placement-discussion-"))
    check(arguments.url, output)


if __name__ == "__main__":
    main()
