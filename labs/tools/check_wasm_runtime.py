"""Exercise real browser-Python execution in full-page lab notebooks.

Serve a completed site, then run this tool with --base-url. A successful HTTP
response or screenshot is never considered runtime readiness. The checks wait
for notebook controls, submit a prediction, change a causal input, and require
an actual instrument change. Draft exports in Volumes III/IV are also inspected.
Cold Pyodide/CDN installation can take several minutes; --timeout is per phase.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

ROUTES = {
    1: "vol1/lab_02_system_paradigms/",
    2: "vol2/lab_01_introduction/",
    3: "vol3/lab_01_introduction/",
    4: "vol4/lab_01_boundary/",
}


async def settled(page, frame) -> None:
    # Import and cell errors can appear after the controls first mount.
    python_errors = [error for error in getattr(page, "_lab_errors", []) if error["phase"] == "Python execution"]
    assert not python_errors, python_errors
    await page.wait_for_timeout(600)


async def instruments(frame) -> str:
    # Control labels and selected option text cannot satisfy this comparison.
    return "\n".join(await frame.locator("table:visible, .js-plotly-plot:visible, marimo-table:visible").all_text_contents())


async def changed_instruments(page, frame, before: str, timeout: int) -> str:
    deadline = asyncio.get_running_loop().time() + timeout / 1000
    while asyncio.get_running_loop().time() < deadline:
        await settled(page, frame)
        after = await instruments(frame)
        if after.strip() and after != before:
            return after
    raise AssertionError("The causal input changed but the rendered instrument did not update")


async def exercise(page, volume: int, base: str, output: Path, timeout: int) -> dict:
    print(f"Volume {volume}: opening full-page notebook and waiting for Python controls", flush=True)
    await page.goto(base.rstrip("/") + "/" + ROUTES[volume], wait_until="domcontentloaded")
    await page.wait_for_url("**/app/index.html*", timeout=timeout)
    await page.get_by_role("navigation", name="Lab course navigation").wait_for(timeout=timeout)
    assert await page.locator("iframe.lab-app").count() == 0
    frame = page
    deadline = asyncio.get_running_loop().time() + timeout / 1000
    polls = 0
    while not await frame.get_by_role("radio").first.is_visible():
        await settled(page, frame)
        if await frame.get_by_role("radio").first.is_visible():
            break
        polls += 1
        if polls % 20 == 0:
            diagnostics = output / "diagnostics"
            diagnostics.mkdir(exist_ok=True)
            body = await frame.locator("body").inner_text()
            (diagnostics / f"volume-{volume}-loading.txt").write_text(body)
            await page.screenshot(path=str(diagnostics / f"volume-{volume}-loading.png"))
            print(f"Volume {volume}: loading state: {body[:100]!r}", flush=True)
        # Marimo virtualizes offscreen cell outputs. The learning map can occupy
        # the opening viewport; scroll the real app surfaces as a student does.
        await frame.locator("body").evaluate("""() => {
            let moved = false;
            const roots = [...new Set([document.scrollingElement, ...document.querySelectorAll('*')])];
            for (const el of roots) {
                if (!el) continue;
                const style = getComputedStyle(el);
                if ((el === document.scrollingElement || /(auto|scroll)/.test(style.overflowY)) && el.scrollHeight > el.clientHeight + 300) {
                    const before = el.scrollTop;
                    el.scrollTop += 350;
                    moved ||= el.scrollTop !== before;
                }
            }
            if (!moved) {
                window.scrollTo(0, 0);
                for (const el of roots) if (el) el.scrollTop = 0;
            }
        }""")
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("Python notebook controls did not mount before the cold-start deadline")
    await settled(page, frame)
    page._lab_runtime_ready = True
    print(f"Volume {volume}: Python controls mounted", flush=True)
    if volume in (1, 2):
        # Exercise a controlled change in the selected project.
        await frame.get_by_role("radio", name=("Payload reduction" if volume == 1 else "More machines make the original state fit"), exact=True).check()
        if volume == 2:
            await frame.get_by_role("button", name="Lock Part A prediction", exact=True).click()
            await frame.get_by_text("Observed consequence:", exact=False).wait_for(timeout=timeout)
            before = await instruments(frame)
            await frame.get_by_role("radio", name="Reduce state on each machine", exact=True).check()
        else:
            lock = frame.get_by_role("button", name="Lock A prediction", exact=True)
            if await lock.is_visible():
                await lock.click()
            await frame.get_by_text("Observed:", exact=False).first.wait_for(timeout=timeout)
            print("Volume 1: prediction feedback opened", flush=True)
            before = await instruments(frame)
            control = frame.get_by_label("Request payload", exact=True)
            options = await control.locator("option").evaluate_all("els => els.filter(e=>!e.disabled).map(e=>e.value)")
            value = await control.input_value()
            await control.select_option(next(item for item in options if item != value))
    elif volume == 3:
        active = frame.locator('[role="tabpanel"]:visible').first
        # Predictions are chapter-specific; do not require an obsolete generic
        # answer string to exercise the actual prediction gate.
        await active.get_by_role("radio").first.check()
        trial = active.get_by_label("Choose one policy", exact=True).first
        await trial.wait_for(timeout=timeout)
        options = await trial.locator("option").evaluate_all("els=>els.filter(e=>!e.disabled && e.value && e.value !== '--').map(e=>e.value)")
        assert len(options) > 1, "The study must expose contrasting policies"
        await trial.select_option(options[0])
        await active.locator("caption:visible").filter(has_text="Supplied baseline and selected policy").first.wait_for(timeout=timeout)
        before = await instruments(frame)
        await trial.select_option(options[1])
    else:
        await frame.get_by_role("radio").first.check()
        await frame.get_by_text("Live evidence and feedback", exact=False).wait_for(timeout=timeout)
        before = await instruments(frame)
        trial = frame.get_by_label("Choose a case to investigate", exact=True).first
        options = await trial.locator("option").evaluate_all("els=>els.filter(e=>!e.disabled).map(e=>e.value)")
        value = await trial.input_value()
        await trial.select_option(next(item for item in options if item != value))
    print(f"Volume {volume}: causal input changed; waiting for instrument", flush=True)
    after = await changed_instruments(page, frame, before, timeout)
    result = {"volume": volume, "route": ROUTES[volume], "scope": "Part A runtime smoke; not a complete lab submission", "causal_instrument_update": True,
              "before": before, "after": after, "download": "not exercised"}
    if volume in (3, 4):
        name = "Download discussion draft" if volume == 3 else "Download discussion"
        async with page.expect_download(timeout=timeout) as event:
            await frame.get_by_role("link", name=name, exact=True).click()
        download = await event.value
        path = output / download.suggested_filename
        await download.save_as(path)
        content = path.read_text()
        assert content.strip(), "Downloaded evidence must contain content"
        if volume == 4:
            assert "Status: incomplete discussion draft" in content, "Unsaved physical work must be labeled as a draft"
        if volume == 3:
            assert "Draft; some comparisons are still missing" in content, "Unsaved agentic work must retain draft status"
            assert "Parts still needing evidence:" in content
            await frame.get_by_text("Full evidence JSON", exact=True).click()
            async with page.expect_download(timeout=timeout) as evidence_event:
                await frame.get_by_role("link", name="Download full evidence JSON", exact=True).click()
            evidence_download = await evidence_event.value
            evidence_path = output / evidence_download.suggested_filename
            await evidence_download.save_as(evidence_path)
            payload = json.loads(evidence_path.read_text())
            assert payload["chapter"] == 1 and payload["missing_parts"], "Draft must honestly retain incomplete status"
            result["evidence_download"] = str(evidence_path)
        result["download"] = str(path)
    await settled(page, frame)
    (output / f"volume-{volume}-runtime.txt").write_text(await frame.locator("body").inner_text())
    await page.screenshot(path=str(output / f"volume-{volume}-runtime.png"), full_page=True)
    for width in (390, 1440):
        await page.set_viewport_size({"width": width, "height": 1000})
        await page.wait_for_timeout(300)
        layout = await page.evaluate("""() => {
            const nav = document.querySelector('#lab-site-chrome');
            const root = document.querySelector('#root');
            return {navBottom:nav.getBoundingClientRect().bottom,
                    rootTop:root.getBoundingClientRect().top,
                    rootBottom:root.getBoundingClientRect().bottom,
                    viewport:innerHeight, overflow:document.documentElement.scrollWidth-innerWidth,
                    navVisible:nav.contains(document.elementFromPoint(20, 20))};
        }""")
        assert abs(layout['rootTop'] - layout['navBottom']) < 2, layout
        assert abs(layout['rootBottom'] - layout['viewport']) < 2, layout
        assert layout['overflow'] <= 1 and layout['navVisible'], layout
        await page.screenshot(path=str(output / f"volume-{volume}-fullpage-{width}.png"), full_page=True)
    if volume == 1:
        # Shared-site choice must survive navigation and native notebook startup.
        toggle = page.locator('#quarto-header .quarto-color-scheme-toggle')
        await toggle.click()
        await page.wait_for_function("document.documentElement.dataset.bsTheme === 'dark' && document.body.classList.contains('dark')")
        assert await page.evaluate("localStorage.getItem('quarto-color-scheme')") == 'alternate'
        await page.screenshot(path=str(output / 'shared-header-dark.png'), full_page=True)
        await page.reload(wait_until='domcontentloaded')
        await page.locator('.nb-header').wait_for(timeout=timeout)
        await page.wait_for_function("document.documentElement.dataset.bsTheme === 'dark' && document.body.classList.contains('dark')")
        await page.screenshot(path=str(output / 'shared-header-dark-after-reload.png'), full_page=True)
        await toggle.click()
        await page.wait_for_function("document.documentElement.dataset.bsTheme === 'light' && document.body.classList.contains('light')")
        result['shared_theme_persistence'] = True
    result['fullpage_desktop_mobile'] = True
    return result


async def main(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(viewport={"width": 1440, "height": 1100}, accept_downloads=True)
        # Reuse the browser cache for common runtime dependencies across volumes.
        results = []
        for volume in args.volumes:
            page = await context.new_page()
            page_errors = []
            page._lab_errors = page_errors
            page._lab_runtime_ready = False
            page.on("pageerror", lambda error: page_errors.append({"message": str(error), "phase": "interaction" if page._lab_runtime_ready else "cold start"}))
            def record_python_error(message):
                content = message.text
                if "[STDERR]" in content and ("Traceback" in content or '"type":"exception"' in content):
                    page_errors.append({"message": content, "phase": "Python execution"})
            page.on("console", record_python_error)
            try:
                result = await exercise(page, volume, args.base_url, args.output, args.timeout * 1000)
                fatal = [error for error in page_errors if error["message"] != "RPC request timed out." or error["phase"] != "cold start"]
                assert not fatal, fatal
                result["startup_warnings"] = page_errors
                results.append(result)
                (args.output / f"volume-{volume}-result.json").write_text(json.dumps(result, indent=2))
                print(f"Volume {volume}: real WASM prediction/input/instrument passed; download={result['download']}", flush=True)
            except Exception:
                bodies = [await item.locator("body").inner_text() for item in page.frames]
                (args.output / f"volume-{volume}-failure.txt").write_text("\n\n--- frame ---\n\n".join(bodies))
                await page.screenshot(path=str(args.output / f"volume-{volume}-failure.png"), full_page=True)
                raise
            finally:
                await page.close()
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:29418")
    parser.add_argument("--output", type=Path, default=Path("/tmp/mlsysbook-wasm-runtime-qa"))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--volumes", nargs="+", type=int, choices=ROUTES, default=list(ROUTES))
    asyncio.run(main(parser.parse_args()))
