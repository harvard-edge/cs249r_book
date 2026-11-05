"""Behavioral checks for independent incident and training-copy lifetimes."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "books/_extensions/mlsysbook-ext/figure-explore"
ART = ROOT / "books/vol4/17_responsibility/images/svg/incident_evidence_lifecycle.svg"
NAME = "original-v4-17-incident-evidence-lifecycle"


def test_incident_and_training_copies_have_separate_lifetimes():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(reduced_motion="reduce")
        page.set_content('<div id="outer" data-mlsys-movement-ms="5000" data-mlsys-reading-pause-ms="100"><div id="root"></div></div>')
        page.add_style_tag(path=str(EXT / "original-player.css"))
        page.add_style_tag(path=str(EXT / (NAME + ".css")))
        page.add_script_tag(path=str(EXT / "original-player.js"))
        page.add_script_tag(path=str(EXT / (NAME + ".js")))
        page.evaluate("""async ([name, art]) => {
            const svg = new DOMParser().parseFromString(art, 'image/svg+xml').documentElement;
            window.player = await window.MLSysFigureLessons[name](document.querySelector('#root'), {
                outer: document.querySelector('#outer'), verifiedArtwork: svg
            });
        }""", [NAME, ART.read_text()])

        def state_at(index):
            page.evaluate("index => player.go(index)", index)
            return json.loads(page.locator("#root").get_attribute("data-incident-state"))

        captured = state_at(2)
        assert captured["routine"] == ["B", "C"]
        assert captured["eventSelected"] and not captured["incidentRetained"]
        retained = state_at(3)
        assert retained["routine"] == ["B", "C"] and retained["incidentRetained"]
        rolled_off = state_at(4)
        assert rolled_off["routine"] == [] and rolled_off["incidentRetained"]
        assert not rolled_off["trainingRetained"]
        pending = state_at(5)
        assert pending["trainingReview"] == "pending" and not pending["trainingRetained"]
        assert page.locator('svg [data-record="?"]').count() == 1
        assert page.locator('svg [data-record="E"][data-deleted="false"]').count() == 1
        approved = state_at(6)
        assert approved["incidentRetained"] and approved["trainingRetained"]
        incident_expired = state_at(7)
        assert incident_expired["incidentDeleted"] and not incident_expired["incidentRetained"]
        assert incident_expired["trainingRetained"] and not incident_expired["trainingDeleted"]
        all_expired = state_at(8)
        assert all_expired["trainingDeleted"] and not all_expired["trainingRetained"]
        assert page.locator('svg [data-record="E"][data-deleted="false"]').count() == 0

        # The response is still C during transport; event E exists only on arrival.
        state_at(1)
        page.emulate_media(reduced_motion="no-preference")
        page.evaluate("player.next()")
        page.wait_for_selector('[data-transient="true"]')
        page.evaluate("player.pause()")
        assert page.locator('[data-transient="true"] text').text_content() == "C"
        assert not json.loads(page.locator('#root').get_attribute('data-incident-state'))['eventSelected']
        page.evaluate("player.restart()")
        assert json.loads(page.locator('#root').get_attribute('data-incident-state'))['routine'] == ['A', 'B']
        assert page.locator('[data-transient="true"]').count() == 0
        page.evaluate("player.destroy()")
        browser.close()
