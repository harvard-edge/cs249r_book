"""Audit canonical labs in native Python or an exported WASM site.

Run with ``--volume 1 --output /tmp/labs-qa`` for native execution, or add
``--base-url http://localhost:29418`` for actual browser Python execution.
Each concept retains prediction submission, active-tab stability, substantive
instrument before/after snapshots, capture actions, errors, and screenshots.

This is one default project per notebook, not all project permutations or a
student-learning evaluation. Orientation gets interface/save/reload/download
checks instead of a fabricated numerical-change requirement. Native runtime
home directories are isolated under the report; user ledgers are never read.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright
from interaction_lab_smoke import (
    _safe_marimo_code,
    safe_marimo_env,
    wait_for_server,
    page_text,
)

ROOT = Path(__file__).resolve().parents[2]


def vis(loc):
    return [loc.nth(i) for i in range(loc.count()) if loc.nth(i).is_visible()]


def pause(p):
    p.wait_for_timeout(1000)


def evidence(p):
    # Compare consequences, not UI labels, styling, or reordered comparison rows.
    return p.locator(
        "table:visible,marimo-table:visible,.js-plotly-plot:visible,marimo-callout-output:visible,.failure-card:visible,.success-card:visible"
    ).evaluate_all(r"""els=>{const records=els.map(e=>{
      if(e.classList.contains('js-plotly-plot')) {
        const traces=(e.data||[]).map(t=>Object.fromEntries(Object.entries(t).filter(([k])=>!['name','marker','line','uid','legendgroup','showlegend','hovertemplate'].includes(k))));
        return {plot:traces.sort((a,b)=>JSON.stringify(a).localeCompare(JSON.stringify(b))),shapes:e.layout?.shapes};
      }
      if(e.matches('.failure-card,.success-card'))return {consequence:e.innerText};
      if(e.tagName.toLowerCase()==='marimo-callout-output') {
        let text=(e.shadowRoot?.textContent||e.innerText||'').replace(/\s+/g,' ').trim();
        text=text.replace(/Your prediction:[^.]+\.\s*/i,'').replace(/Selected candidate:[^·]+·/i,'Candidate verdict:').replace(/Constrained ranking recommends (.*?) for .*? under/i,'Constrained ranking recommends $1 under').replace(/Compared live result:\s*\S+\s+detects/i,'Compared live result detects');
        if(!/Observed|Result:|verdict:|Selected evidence|Constrained ranking recommends|Artifact version|detects|Release gate|speedup|throughput|quality|latency|margin|Decision consequence|selected topology|plan follows|communication|\bFAIL\b|\bPASS\b|\bHOLD\b/i.test(text))return null;
        return {consequence:text};
      }
      const rows=[...e.querySelectorAll('tr')].map(r=>[...r.querySelectorAll('th,td')].map(c=>c.innerText));
      if(!rows.length)return {text:e.innerText};
      const headers=rows.shift();const keep=headers.map((h,i)=>/^(policy|case|choice|selected policy|run|setting)$/i.test(h.trim())?-1:i).filter(i=>i>=0);
      const values=rows.map(r=>keep.map(i=>r[i]));
      if(!headers.some(h=>/step|time|event|sequence|order/i.test(h)))values.sort((a,b)=>JSON.stringify(a).localeCompare(JSON.stringify(b)));
      return {headers:keep.map(i=>headers[i]),rows:values};
    }).filter(x=>x!==null);return records.length?JSON.stringify(records):'';}""")


def unlock(p):
    groups = vis(p.get_by_role("radiogroup"))
    if groups:
        for group in groups:
            radios = vis(group.get_by_role("radio"))
            if radios and not any(r.is_checked() for r in radios):
                radios[0].check()
                pause(p)
    else:
        radios = vis(p.get_by_role("radio"))
        if radios and not any(r.is_checked() for r in radios):
            radios[0].check()
            pause(p)
    submitted = []
    for button in vis(
        p.get_by_role(
            "button", name=re.compile("^(Lock .*prediction|Record prediction)$")
        )
    ):
        if button.is_enabled():
            submitted.append(button.inner_text())
            button.click()
            pause(p)
    return {"submitted_forms": submitted, "radio_groups": len(groups)}


def show_traces(p):
    for button in vis(
        p.get_by_role(
            "button",
            name=re.compile(
                "Inspect the selected event trace|Exact study records", re.I
            ),
        )
    ):
        if button.get_attribute("aria-expanded") == "false":
            button.click()
            pause(p)


def compare(p, before):
    for b in vis(
        p.get_by_role("button", name=re.compile("^(Compare|Run |Apply )", re.I))
    ):
        if b.is_enabled():
            b.click()
            pause(p)
    show_traces(p)
    deadline = time.monotonic() + 3
    after = evidence(p)
    while (not after.strip() or after == before) and time.monotonic() < deadline:
        p.wait_for_timeout(300)
        after = evidence(p)
    return {
        "changed": bool(before.strip() and after.strip() and before != after),
        "before": before,
        "after": after,
    }


def change(p):
    show_traces(p)
    attempts = []
    controls = [
        (
            c.evaluate(
                "e=>e.parentElement.querySelector('label')?.textContent||e.getAttribute('aria-label')||''"
            ),
            c,
        )
        for c in vis(p.locator("select"))
    ]
    controls.sort(
        key=lambda item: (
            0
            if re.search(
                "change one|intervention|policy|payload|rate|memory", item[0], re.I
            )
            else 1
        )
    )
    for label, c in controls:
        if re.search("track|project|deployment|training context", label, re.I):
            continue
        # A dependent selector can unmount/reorder after another trial changes.
        # Rebind through its public label rather than a stale nth(index).
        if label:
            c = p.get_by_label(label, exact=True).first
            if not c.count():
                continue
        old = c.input_value()
        opts = c.locator("option").evaluate_all(
            'es=>es.filter(e=>!e.disabled&&e.value&&e.value!=="--").map(e=>({value:e.value,label:e.textContent}))'
        )
        for o in opts:
            if o["value"] == old:
                continue
            if not c.count() or not c.locator("option").evaluate_all(
                "(es,v)=>es.some(e=>e.value===v)", o["value"]
            ):
                continue
            before = evidence(p)
            c.select_option(o["value"])
            pause(p)
            r = compare(p, before)
            attempts.append(
                {"control": label, "choice": o["label"], "changed": r["changed"]}
            )
            if r["changed"]:
                return r | {"attempts": attempts}
    for c in vis(p.locator("input[type=range],input[type=number]")):
        old = c.input_value()
        maximum = c.get_attribute("max")
        step = c.get_attribute("step") or "1"
        try:
            new = str(
                float(maximum)
                if maximum and float(maximum) != float(old)
                else float(old) + float(step)
            )
        except ValueError:
            continue
        before = evidence(p)
        c.fill(new)
        c.dispatch_event("input")
        c.dispatch_event("change")
        pause(p)
        r = compare(p, before)
        attempts.append(
            {
                "control": c.get_attribute("aria-label"),
                "choice": new,
                "changed": r["changed"],
            }
        )
        if r["changed"]:
            return r | {"attempts": attempts}
    for c in vis(p.get_by_role("combobox")):
        if c.evaluate("e=>e.tagName") == "SELECT":
            continue
        before = evidence(p)
        c.click()
        pause(p)
        opts = vis(p.get_by_role("option"))
        if opts:
            opts[0].click()
            p.keyboard.press("Escape")
            pause(p)
            r = compare(p, before)
            attempts.append({"multiselect": True, "changed": r["changed"]})
        else:
            p.keyboard.press("Escape")
            continue
        if r["changed"]:
            return r | {"attempts": attempts}
    for c in vis(p.get_by_role("slider")):
        for direction in ("End", "Home"):
            before = evidence(p)
            c.focus()
            c.press(direction)
            pause(p)
            r = compare(p, before)
            attempts.append(
                {
                    "slider": c.get_attribute("aria-label"),
                    "direction": direction,
                    "changed": r["changed"],
                }
            )
            if r["changed"]:
                return r | {"attempts": attempts}
    for c in vis(p.get_by_role("checkbox")):
        before = evidence(p)
        c.set_checked(not c.is_checked())
        pause(p)
        r = compare(p, before)
        attempts.append(
            {"checkbox": c.get_attribute("aria-label"), "changed": r["changed"]}
        )
        if r["changed"]:
            return r | {"attempts": attempts}
    for c in vis(p.get_by_role("radio")):
        if c.is_checked():
            continue
        before = evidence(p)
        c.check()
        pause(p)
        r = compare(p, before)
        attempts.append({"radio": c.get_attribute("value"), "changed": r["changed"]})
        if r["changed"]:
            return r | {"attempts": attempts}
    return {"changed": False, "attempts": attempts, "instrument": evidence(p)}


def prepare_part(p):
    for button in vis(
        p.get_by_role(
            "button", name=re.compile("Help me design a comparison|Prediction", re.I)
        )
    ):
        if button.get_attribute("aria-expanded") == "false":
            button.click()
            pause(p)


def capture_part(p):
    actions = []
    # Reveal explicitly supplied labels when a concept depends on observing them.
    for field in vis(p.get_by_role("checkbox", name=re.compile("Reveal.*label", re.I))):
        if not field.is_checked():
            field.check()
            pause(p)
    for field in vis(p.get_by_role("textbox")):
        if field.evaluate("e=>e.tagName") == "TEXTAREA":
            field.fill(
                "The changed case alters the displayed obligation under the supplied assumptions. I would revisit this choice if the task boundary or measured evidence changes."
            )
            pause(p)
    unlock(p)
    for button in vis(
        p.get_by_role(
            "button",
            name=re.compile(
                "^(Capture |Save (?!discussion)|Attach [A-E] interpretation)", re.I
            ),
        )
    ):
        if button.is_enabled():
            actions.append(button.inner_text())
            button.click()
            pause(p)
    # Some interpretation fields mount only after a contrast is frozen.
    for field in vis(p.get_by_role("textbox")):
        if (
            field.evaluate("e=>e.tagName") == "TEXTAREA"
            and not field.input_value().strip()
        ):
            field.fill(
                "The comparison changes the displayed requirement; the bounded study does not validate deployment. Revisit when the workload or evidence changes."
            )
            pause(p)
    for button in vis(
        p.get_by_role("button", name=re.compile("^Attach [A-E] interpretation"))
    ):
        if button.is_enabled():
            actions.append(button.inner_text())
            button.click()
            pause(p)
    return actions


def orientation_workflow(p, directory):
    checks = []

    def tab(letter):
        p.get_by_role("tab", name=re.compile("^" + letter + " ·")).click()
        pause(p)

    def snapshot(letter):
        for width in (1440, 390):
            p.set_viewport_size({"width": width, "height": 1000})
            pause(p)
            p.get_by_role("tabpanel").first.scroll_into_view_if_needed()
            p.screenshot(path=str(directory / f"part-{letter}-{width}.png"))
        p.set_viewport_size({"width": 1440, "height": 1000})

    tab("A")
    control = p.get_by_label("Compare with", exact=True)
    before = p.locator(".orientation-card").all_text_contents()
    choices = control.locator("option").evaluate_all("es=>es.map(e=>e.value)")
    control.select_option(next(x for x in choices if x != control.input_value()))
    pause(p)
    after = p.locator(".orientation-card").all_text_contents()
    assert before != after, "Project briefing did not update"
    snapshot("A")
    checks.append(
        {
            "part": "A",
            "mode": "orientation workflow",
            "project_brief_changes": True,
            "passed": True,
        }
    )
    tab("B")
    p.get_by_role("radio").first.check()
    p.get_by_role("button", name="Lock my prediction", exact=True).click()
    pause(p)
    case = p.get_by_label("Live case", exact=True)
    case.select_option(label="Alternative briefing")
    pause(p)
    p.get_by_role("button", name="Capture this comparison", exact=True).click()
    pause(p)
    case.select_option(label="Baseline briefing")
    pause(p)
    frozen = p.locator(".orientation-saved:visible").inner_text()
    assert "Saved case: alternative" in frozen and "Live case: baseline" in frozen, (
        "Live change rewrote retained comparison"
    )
    snapshot("B")
    checks.append(
        {
            "part": "B",
            "mode": "orientation workflow",
            "prediction_submitted": True,
            "live_change_keeps_frozen_comparison": True,
            "passed": True,
        }
    )
    tab("C")
    p.get_by_role("button", name="Save my progress", exact=True).click()
    p.get_by_text("Saved and read back.", exact=False).first.wait_for(timeout=30000)
    with p.expect_download() as event:
        p.get_by_role("link", name="Download your work", exact=True).click()
    download = event.value
    path = directory / "orientation-download.json"
    download.save_as(path)
    payload = json.loads(path.read_text())
    assert payload["comparison"]["result"]["inputs"]["case"] == "alternative"
    snapshot("C")
    p.reload()
    p.get_by_role("tab").first.wait_for(timeout=240000)
    tab("C")
    p.get_by_text("Restored from an earlier visit.", exact=False).first.wait_for(
        timeout=30000
    )
    checks.append(
        {
            "part": "C",
            "mode": "orientation workflow",
            "explicit_save_readback": True,
            "download_matches_capture": True,
            "reload_restores": True,
            "passed": True,
        }
    )
    tab("D")
    link = p.get_by_role("link", name=re.compile("Open Lab 01"))
    assert link.count()
    resolved = link.evaluate("e=>e.href")
    if "/app/" in p.url:
        assert (
            "/vol1/lab_01_ml_intro/" in resolved
            and "/lab_00_introduction/" not in resolved
        ), resolved
    snapshot("D")
    checks.append(
        {
            "part": "D",
            "mode": "orientation workflow",
            "next_lab_handoff_present": True,
            "passed": True,
        }
    )
    return checks


def audit(browser, c, out, port, base=None):
    lab = ROOT / c["notebook"]
    d = out / f"v{c['volume']}_{c['chapter']:02}"
    d.mkdir(parents=True, exist_ok=True)
    sandbox = d / "runtime"
    sandbox.mkdir(exist_ok=True)
    r = {
        "lab": c["notebook"],
        "mode": "wasm" if base else "native",
        "comparison_version": "semantic-outcome-and-trace-v2",
        "source_sha256": hashlib.sha256(lab.read_bytes()).hexdigest(),
        "parts": [],
        "errors": [],
    }
    proc = None
    log = None
    p = browser.new_page()
    p.set_viewport_size({"width": 1440, "height": 1000})
    p.on("pageerror", lambda e: r["errors"].append(str(e)))
    p.set_default_timeout(6000)
    try:
        if base:
            url = (
                base.rstrip("/")
                + "/"
                + str(Path(c["notebook"]).relative_to("labs").with_suffix(""))
                + "/"
            )
        else:
            code = _safe_marimo_code(lab, port).replace(
                "sys.argv =",
                f"import pathlib\npathlib.Path.home=classmethod(lambda cls:pathlib.Path({str(sandbox)!r}))\nsys.argv =",
            )
            # Marimo can spawn a separate Python kernel. An inherited startup
            # hook isolates that child too; patching the CLI alone is insufficient.
            env = safe_marimo_env()
            startup = Path(env["PYTHONPATH"].split(":", 1)[0]) / "sitecustomize.py"
            (sandbox / "sitecustomize.py").write_text(
                startup.read_text()
                + "\nimport os, pathlib\npathlib.Path.home=classmethod(lambda cls:pathlib.Path(os.environ['MLSYSBOOK_QA_RUNTIME_HOME']))\n"
            )
            env["MLSYSBOOK_QA_RUNTIME_HOME"] = str(sandbox)
            env["PYTHONPATH"] = str(sandbox) + ":" + env["PYTHONPATH"]
            log = (d / "server.log").open("w")
            proc = subprocess.Popen(
                [sys.executable, "-c", code],
                cwd=sandbox,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            url = f"http://localhost:{port}"
            wait_for_server(url, 60)
        p.goto(url, wait_until="domcontentloaded")
        p.get_by_role("tab").first.wait_for(timeout=240000)
        pause(p)
        for width in (1440, 390):
            p.set_viewport_size({"width": width, "height": 1000})
            p.evaluate(
                "() => {for(const e of document.querySelectorAll('*'))if(e.scrollTop)e.scrollTop=0;window.scrollTo(0,0)}"
            )
            pause(p)
            p.screenshot(path=str(d / f"opening-{width}.png"))
        p.set_viewport_size({"width": 1440, "height": 1000})
        pause(p)
        if c["volume"] == 1 and c["chapter"] == 0:
            r["parts"] = orientation_workflow(p, d)
        for part in [] if c["volume"] == 1 and c["chapter"] == 0 else c["parts"]:
            q = {
                "part": part["id"],
                "reachable": False,
                "prediction_opened": False,
                "causal_change": False,
            }
            try:
                tabs = p.get_by_role("tab")
                matches = [
                    tabs.nth(i)
                    for i in range(tabs.count())
                    if re.match(
                        rf"^(Part {part['id']}\b|{part['id']}\s*[·.])",
                        tabs.nth(i).inner_text().strip(),
                    )
                ]
                if not matches:
                    raise RuntimeError("No concept tab matching contract")
                matches[0].click()
                pause(p)
                q["reachable"] = True
                expected_tab = matches[0].inner_text().strip()
                prepare_part(p)
                unlock_record = unlock(p)
                pause(p)
                q["prediction_unlock"] = unlock_record
                q["active_tab_after_prediction"] = (
                    p.get_by_role("tab", selected=True).first.inner_text().strip()
                )
                if q["active_tab_after_prediction"] != expected_tab:
                    raise AssertionError(
                        "Prediction submission reset the active concept tab"
                    )
                q["prediction_opened"] = bool(unlock_record["submitted_forms"]) or any(
                    x.is_checked() for x in vis(p.get_by_role("radio"))
                )
                for button in vis(p.get_by_role("button", name=re.compile("^Pin "))):
                    if button.is_enabled():
                        button.click()
                        pause(p)
                ledger_workflow = (
                    c["volume"] == 1 and c["chapter"] == 16 and part["id"] == "A"
                )
                trial = (
                    {
                        "changed": False,
                        "not_applicable": "Inspect the real saved ledger; do not fabricate a control that rewrites history.",
                    }
                    if ledger_workflow
                    else change(p)
                )
                q["trial"] = trial
                q["active_tab_after_trial"] = (
                    p.get_by_role("tab", selected=True).first.inner_text().strip()
                )
                if q["active_tab_after_trial"] != expected_tab:
                    raise AssertionError(
                        "Causal intervention reset the active concept tab"
                    )
                q["causal_change"] = trial["changed"]
                q["capture_actions"] = capture_part(p)
                if ledger_workflow:
                    q["investigation_mode"] = "actual ledger inspection workflow"
                    q["workflow_verified"] = (
                        "Capture ledger contrast" in q["capture_actions"]
                        and all(
                            word in evidence(p)
                            for word in ("Chapter", "Status", "Reason")
                        )
                        and "Missing evidence is not a failed requirement"
                        in page_text(p)
                        and "Saved snapshot" in page_text(p)
                    )
                q["active_tab_after_capture"] = (
                    p.get_by_role("tab", selected=True).first.inner_text().strip()
                )
                if q["active_tab_after_capture"] != expected_tab:
                    raise AssertionError(
                        "Evidence capture reset the active concept tab"
                    )
                body = page_text(p)
                q["runtime_errors"] = re.findall(
                    "Traceback|ModuleNotFoundError|NameError:|An exception was raised",
                    body,
                )
                (d / f"part-{part['id']}.txt").write_text(body)
                ins = p.locator(".js-plotly-plot:visible").first
                if not ins.count():
                    ins = p.locator("table:visible").last
                if ins.count():
                    ins.scroll_into_view_if_needed()
                q["desktop_instrument_rows"] = (
                    ins.locator("tr").count() if ins.count() else 0
                )
                p.screenshot(path=str(d / f"part-{part['id']}-desktop.png"))
                p.set_viewport_size({"width": 390, "height": 1000})
                pause(p)
                if ins.count():
                    ins.scroll_into_view_if_needed()
                pause(p)
                q["mobile_instrument_rows"] = (
                    ins.locator("tr").count() if ins.count() else 0
                )
                if q["desktop_instrument_rows"] > 1:
                    deadline = time.monotonic() + 3
                    while (
                        q["mobile_instrument_rows"] < 2 and time.monotonic() < deadline
                    ):
                        p.wait_for_timeout(300)
                        q["mobile_instrument_rows"] = ins.locator("tr").count()
                    if q["mobile_instrument_rows"] < 2:
                        raise AssertionError(
                            "The visible evidence table lost its data rows on mobile"
                        )
                q["mobile_overflow"] = p.evaluate(
                    "Math.max(0,document.documentElement.scrollWidth-innerWidth)"
                )
                p.screenshot(path=str(d / f"part-{part['id']}-mobile.png"))
                p.set_viewport_size({"width": 1440, "height": 1000})
                pause(p)
            except Exception as e:
                q["failure"] = str(e)
                try:
                    p.screenshot(path=str(d / f"part-{part['id']}-failure.png"))
                    (d / f"part-{part['id']}-failure.txt").write_text(page_text(p))
                except Exception:
                    pass
            q["passed"] = bool(
                q["reachable"]
                and q["prediction_opened"]
                and (q["causal_change"] or q.get("workflow_verified", False))
                and not q.get("runtime_errors")
                and not q.get("failure")
                and q.get("mobile_overflow", 999) <= 2
            )
            r["parts"].append(q)
            (d / "result.json").write_text(json.dumps(r, indent=2))
        r["passed"] = all(q["passed"] for q in r["parts"]) and not r["errors"]
    except Exception as e:
        r["failure"] = str(e)
        r["passed"] = False
    finally:
        p.close()
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if log:
            log.close()
        (d / "result.json").write_text(json.dumps(r, indent=2))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True, type=int)
    ap.add_argument("--chapters", nargs="*", type=int)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--port", type=int, default=37631)
    ap.add_argument("--base-url")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cs = json.loads(
        (
            ROOT / f"labs/mlsysbook_labs/data/learning_paths/vol{a.volume}.json"
        ).read_text()
    )["chapters"]
    results = []
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        context = b.new_context(accept_downloads=True)
        for c in cs:
            if a.chapters is not None and c["chapter"] not in a.chapters:
                continue
            print(f"auditing v{a.volume}/{c['chapter']:02}", flush=True)
            r = audit(context, c, a.output, a.port, a.base_url)
            results.append(r)
            (a.output / "matrix.json").write_text(json.dumps(results, indent=2))
            print(
                f" {sum(q['passed'] for q in r['parts'])}/{len(r['parts'])} parts",
                flush=True,
            )
        context.close()
        b.close()


if __name__ == "__main__":
    main()
