"""Check every concept map on the taxonomy explorer for overlapping edge labels.

Renders each concept page in Chrome, reads label and node boxes from the
Cytoscape instance, and fails if any edge label overlaps another label or a
node. Added 2026-10-22 after two-link concept maps stacked their verb labels
(107 concepts affected before the layout fix). Needs Playwright with Chrome.

Usage: render site, serve site/_build, then
    python3 site/scripts/check_concept_map_layout.py http://localhost:8000/taxonomy/index.html
"""
from playwright.sync_api import sync_playwright
import sys
from pathlib import Path

import yaml
ROOT = Path(__file__).resolve().parents[2]
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/taxonomy/index.html"
source = yaml.safe_load((ROOT / "taxonomy/taxonomy.yml").read_text())
ids = [c["id"] for c in source["concepts"]]
statements = {cid: sorted((r["predicate"], r.get("conditions", ""))
    for r in source["relationships"] if cid in (r["source"], r["target"])) for cid in ids}
HOOK = """
(() => { let f; Object.defineProperty(window, 'cytoscape', { configurable: true,
  set(v) { f = v; }, get() { if (!f) return undefined; const w = function(...a) { const c = f(...a); window.__cy = c; return c; };
  Object.assign(w, f); return w; } }); })();
"""
MEASURE = """
() => { const cy = window.__cy; if (!cy) return null;
  const boxes = [];
  cy.edges().forEach(e => { const b = e.boundingBox({ includeNodes: false, includeEdges: false, includeLabels: true, includeMainLabels: false, includeSourceLabels: true, includeTargetLabels: true, includeOverlays: false });
    if (b.w > 0 && b.h > 0) boxes.push({ id: e.id(), kind: 'label', b }); });
  cy.nodes().forEach(n => { const b = n.boundingBox({ includeLabels: false, includeOverlays: false }); boxes.push({ id: n.id(), kind: 'node', b }); });
  const hit = (a, c) => a.x1 < c.x2 - 1 && c.x1 < a.x2 - 1 && a.y1 < c.y2 - 1 && c.y1 < a.y2 - 1;
  const bad = [];
  for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
    const A = boxes[i], B = boxes[j]; if (A.kind === 'node' && B.kind === 'node') continue;
    if (hit(A.b, B.b)) bad.push([A.id, B.id]); }
  return { edges: cy.edges().length, bad }; }
"""
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1440, "height": 1000})
    pg.add_init_script(HOOK)
    pg.goto(URL, wait_until="networkidle")
    problems = {}; measured = 0; missing = []; changed_statements = []
    for cid in ids:
        pg.evaluate(f"location.hash = 'concept={cid}'"); pg.wait_for_timeout(120)
        actual = pg.locator(".bok-rels tbody tr").evaluate_all("""rows => rows.map(row => [
            row.querySelector('.bok-verb').textContent,
            row.querySelector('td:last-child').textContent])""")
        if sorted(map(tuple, actual)) != statements[cid]:
            changed_statements.append(cid)
        r = pg.evaluate(MEASURE)
        if r is None:
            missing.append(cid)
            continue
        measured += 1
        if r["bad"]: problems[cid] = r["bad"][:4]
    b.close()
print(f"measured {measured} of {len(ids)} concept maps; {len(problems)} with overlapping labels")
for k, v in list(problems.items())[:15]:
    print(" ", k, v)
if missing:
    print("Maps not measured:", ", ".join(missing[:15]))
print(f"{len(changed_statements)} concepts with altered or missing relationship statements")
if changed_statements:
    print("Statement mismatches:", ", ".join(changed_statements[:15]))
sys.exit(1 if problems or missing or changed_statements or not ids else 0)
