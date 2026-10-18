#!/usr/bin/env python3
"""Audit the dark mode of a built Quarto HTML site.

Serves a built site folder, opens every page in WebKit (Safari's engine) with the
OS in light mode, clicks Quarto's colour-scheme toggle (the path a reader takes),
and checks the dark result.

Why WebKit and why the toggle: a Safari-only bug (a CSS custom property that
referenced itself) left every callout header pale on pale text. Chromium never
showed it, and loading with the OS already dark never exercised the toggle path.

Checks, per page:
  * text contrast >= 3:1 against the effective background (skips SVG and math)
  * callout headers (``details[class*="callout-"] > summary``) are dark
  * fenced code blocks are dark and padded
  * section numbers (``.header-section-number``) reach 4.5:1
  * with ``--figures``: figure images are dimmed rather than pure-white cards

Usage:
    python3 binder/tools/audit/audit_dark_mode.py books/_build/html-vol1
    python3 binder/tools/audit/audit_dark_mode.py books/_build/html-vol1 --glob 'vol1/0[1-3]*/*.html'

Exit status is 1 if any page has an issue or fails to load, so it can gate CI.

Needs: ``pip install playwright`` and ``python3 -m playwright install webkit``.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import socketserver
import sys
import threading
from pathlib import Path

PAGE_JS = r"""
(opts) => {
  const parse = c => { const m = c.match(/rgba?\(([^)]+)\)/); if (!m) return [0,0,0,0];
    const p = m[1].split(',').map(s => parseFloat(s)); return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1]; };
  const over = (t, b) => { const a = t[3]; return [t[0]*a + b[0]*(1-a), t[1]*a + b[1]*(1-a), t[2]*a + b[2]*(1-a), 1]; };
  const lum = c => { const f = v => { v /= 255; return v <= 0.03928 ? v/12.92 : Math.pow((v+0.055)/1.055, 2.4); };
    return 0.2126*f(c[0]) + 0.7152*f(c[1]) + 0.0722*f(c[2]); };
  const contrast = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x,y)+0.05) / (Math.min(x,y)+0.05); };
  const effBg = el => { const chain = []; for (let e = el; e; e = e.parentElement) chain.push(parse(getComputedStyle(e).backgroundColor));
    let bg = [26, 26, 26, 1]; for (let i = chain.length - 1; i >= 0; i--) bg = over(chain[i], bg); return bg; };
  const out = {issues: [], counts: {callouts: 0, code: 0, numbers: 0, figures: 0}};
  const add = o => { if (out.issues.length < 25) out.issues.push(o); out.total = (out.total || 0) + 1; };

  const root = document.querySelector('main') || document.body;
  const seen = new Set();
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let n; (n = walker.nextNode());) {
    const t = n.nodeValue.trim(); if (t.length < 2) continue;
    const el = n.parentElement; if (!el || seen.has(el)) continue; seen.add(el);
    const cs = getComputedStyle(el); if (cs.visibility === 'hidden' || cs.display === 'none') continue;
    const r = el.getBoundingClientRect(); if (!r.width || !r.height) continue;
    if (el.closest('svg, mjx-container, .MathJax, script, style')) continue;
    const bg = effBg(el), fg = over(parse(cs.color), bg), c = contrast(fg, bg);
    if (c < 3.0) add({kind: 'text-contrast', tag: el.tagName, text: t.slice(0, 40), contrast: +c.toFixed(2)});
  }
  document.querySelectorAll('details[class*="callout-"] > summary').forEach(s => {
    out.counts.callouts++;
    if (lum(effBg(s)) > 0.35) add({kind: 'callout-header-light', cls: s.parentElement.className.toString().slice(0, 40), text: s.textContent.trim().slice(0, 30)});
  });
  document.querySelectorAll('div.sourceCode').forEach(d => {
    out.counts.code++;
    const pre = d.querySelector('pre');
    const pad = (pre ? parseFloat(getComputedStyle(pre).paddingLeft) : 0) + parseFloat(getComputedStyle(d).paddingLeft);
    if (lum(effBg(d)) > 0.2) add({kind: 'code-block-light', text: d.textContent.trim().slice(0, 30)});
    else if (pad < 8) add({kind: 'code-block-unpadded', padding: pad, text: d.textContent.trim().slice(0, 30)});
  });
  document.querySelectorAll('.header-section-number').forEach(e => {
    out.counts.numbers++;
    const bg = effBg(e), c = contrast(over(parse(getComputedStyle(e).color), bg), bg);
    if (c < 4.5) add({kind: 'section-number-contrast', contrast: +c.toFixed(2), text: e.textContent});
  });
  if (opts.figures) document.querySelectorAll('figure img, .quarto-figure img').forEach(i => {
    out.counts.figures++;
    if (getComputedStyle(i).filter === 'none' && !i.src.endsWith('margin_stack.svg')) add({kind: 'figure-not-dimmed', src: i.src.split('/').pop().slice(0, 40)});
  });
  return out;
}
"""


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs) -> None:  # one line per asset would bury the report
        pass


def serve(root: Path) -> tuple[socketserver.TCPServer, int]:
    handler = functools.partial(_QuietHandler, directory=str(root))
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="built site folder (contains the .html files)")
    ap.add_argument("--glob", default="**/*.html", help="pages to audit, relative to root")
    ap.add_argument("--figures", action="store_true", help="also require figure images to be dimmed (books)")
    ap.add_argument("--json", type=Path, help="write the full result here")
    ap.add_argument("--width", type=int, default=1400)
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed: pip install playwright && python3 -m playwright install webkit", file=sys.stderr)
        return 2

    root = args.root.resolve()
    pages = sorted(p for p in root.glob(args.glob) if "site_libs" not in p.parts and p.name != "search.html")
    if not pages:
        print(f"no pages match {args.glob} under {root}", file=sys.stderr)
        return 2

    server, port = serve(root)
    results: dict[str, dict] = {}
    failed = 0
    with sync_playwright() as pw:
        browser = pw.webkit.launch()
        for page_path in pages:
            rel = page_path.relative_to(root).as_posix()
            ctx = browser.new_context(color_scheme="light", viewport={"width": args.width, "height": 900})
            page = ctx.new_page()
            try:
                page.goto(f"http://127.0.0.1:{port}/{rel}", wait_until="load", timeout=45000)
                page.wait_for_timeout(1500)
                toggle = page.locator("a.quarto-color-scheme-toggle")
                if toggle.count() == 0:
                    results[rel] = {"skipped": "no colour-scheme toggle"}
                    print(f"skip  {rel} (no toggle)")
                    continue
                toggle.first.click()
                page.wait_for_timeout(800)
                res = page.evaluate(PAGE_JS, {"figures": args.figures})
                results[rel] = res
                n = res.get("total", 0)
                failed += bool(n)
                print(f"{'FAIL' if n else 'ok  '}  {rel}  callouts={res['counts']['callouts']} code={res['counts']['code']} issues={n}")
                for issue in res["issues"][:5]:
                    print(f"        {issue}")
            except Exception as exc:  # a page that cannot load is a failure, not a skip
                failed += 1
                results[rel] = {"error": str(exc)[:120]}
                print(f"ERROR {rel}: {str(exc)[:100]}")
            finally:
                ctx.close()
        browser.close()
    server.shutdown()

    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
    print(f"\n{len(pages)} pages, {failed} with issues")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
