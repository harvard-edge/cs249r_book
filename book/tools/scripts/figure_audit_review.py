#!/usr/bin/env python3
"""
Figure Audit Review Dashboard Generator.

Parses audit markdown reports and generates a self-contained HTML review page.
Uses fig-id as the definitive key throughout — matches id="fig-xxx" in rendered
HTML to find the exact image <img src="..."> for each figure.

Usage:
    python3 book/tools/scripts/figure_audit_review.py
    cd book/quarto && python3 -m http.server 8787
    # Open http://localhost:8787/_build/figure_review.html
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = REPO_ROOT / ".claude" / "_reviews" / "figure_audit"
BUILD_DIR = REPO_ROOT / "book" / "quarto" / "_build"
OUTPUT_HTML = BUILD_DIR / "figure_review.html"


def parse_audit_files() -> list[dict]:
    """Parse all audit markdown files and extract figure entries."""
    entries = []

    for audit_file in sorted(AUDIT_DIR.glob("*_audit.md")):
        chapter = audit_file.stem.replace("_audit", "").replace("_", "/")
        text = audit_file.read_text(encoding="utf-8")

        model_match = re.search(r"Model:\s*(.+)", text)
        model = model_match.group(1).strip() if model_match else "unknown"

        fig_pattern = re.compile(
            r"###\s+Figure\s+\d+:\s+`(fig-[\w-]+)`\s+\((?:Line\s+(\d+).*?in\s+)?(.*?)\)\s*\n"
            r"(.*?)(?=###\s+Figure\s+\d+:|## SUMMARY|---|\Z)",
            re.DOTALL,
        )

        for match in fig_pattern.finditer(text):
            fig_id = match.group(1)
            line_num = int(match.group(2)) if match.group(2) else 0
            qmd_file = match.group(3).strip()
            body = match.group(4)

            def extract_field(label: str) -> str:
                pat = re.compile(
                    rf"\d+\.\s+\*\*{label}\*\*:\s*(.*?)(?=\n\d+\.\s+\*\*|\n---|\Z)",
                    re.DOTALL,
                )
                m = pat.search(body)
                return m.group(1).strip() if m else ""

            visual_desc = extract_field("VISUAL DESCRIPTION")
            caption_match = extract_field("CAPTION MATCH")
            alttext_match = extract_field("ALT-TEXT MATCH")
            prose_match = extract_field("PROSE MATCH")

            verdict_match = re.search(r"\*\*VERDICT\*\*:\s*(PASS|MINOR ISSUE|MISMATCH|ERROR|FIGURE_ISSUE|UNVERIFIED|CANNOT_VERIFY)", body)
            verdict = verdict_match.group(1) if verdict_match else "UNKNOWN"

            fix_match = re.search(
                r"\*\*SUGGESTED FIX\*\*:\s*(.*?)(?=\n###|\n---|\n## |\Z)",
                body, re.DOTALL,
            )
            suggested_fix = fix_match.group(1).strip() if fix_match else ""

            entries.append({
                "fig_id": fig_id,
                "chapter": chapter,
                "line": line_num,
                "qmd_file": qmd_file,
                "visual_description": visual_desc,
                "caption_analysis": caption_match,
                "alttext_analysis": alttext_match,
                "prose_analysis": prose_match,
                "verdict": verdict,
                "suggested_fix": suggested_fix,
                "model": model,
            })

    return entries


def build_image_index() -> dict[str, str]:
    """
    Scan ALL rendered HTML files and build a master index:
      fig-id → image path (relative to _build/)

    This is the single source of truth. The fig-id in the HTML
    (id="fig-xxx" on the div) maps to the <img src="..."> inside it.
    """
    index = {}
    for html_file in BUILD_DIR.rglob("*.html"):
        if html_file.name == "figure_review.html":
            continue
        try:
            content = html_file.read_text(encoding="utf-8")
        except Exception:
            continue

        # Find all: <div id="fig-xxx" ...> ... <img src="path"> patterns
        for m in re.finditer(
            r'<div\s+id="(fig-[\w-]+)"[^>]*class="[^"]*quarto-float[^"]*"[^>]*>'
            r'.*?<img\s+src="([^"]+)"',
            content, re.DOTALL,
        ):
            fig_id = m.group(1)
            img_src = m.group(2)
            img_abs = (html_file.parent / img_src).resolve()
            if img_abs.exists():
                try:
                    index[fig_id] = str(img_abs.relative_to(BUILD_DIR))
                except ValueError:
                    pass
    return index


def find_current_text(fig_id: str, qmd_file: str) -> dict:
    """Extract current fig-cap and fig-alt from the QMD source."""
    # binder outputs paths relative to book/quarto/ (e.g. contents/vol1/...)
    qmd_path = REPO_ROOT / "book" / "quarto" / qmd_file
    if not qmd_path.exists():
        # fallback: try from repo root directly
        qmd_path = REPO_ROOT / qmd_file
    if not qmd_path.exists():
        return {"fig_cap": "", "fig_alt": ""}

    text = qmd_path.read_text(encoding="utf-8")

    # Find the figure div block containing this fig-id
    # The div may span multiple lines, so collect the full attribute block
    lines = text.splitlines()
    cap, alt = "", ""
    for i, line in enumerate(lines):
        if f"#{fig_id}" in line and re.match(r"^:{3,}", line):
            # Collect the full div opening (may span lines until closing })
            block = line
            j = i
            while "}" not in block and j < min(i + 10, len(lines) - 1):
                j += 1
                block += " " + lines[j]

            cap_m = re.search(r'fig-cap="((?:[^"\\]|\\")*)"', block)
            alt_m = re.search(r'fig-alt="((?:[^"\\]|\\")*)"', block)
            if cap_m:
                cap = cap_m.group(1).replace('\\"', '"')
            if alt_m:
                alt = alt_m.group(1).replace('\\"', '"')
            break

    return {"fig_cap": cap, "fig_alt": alt}


def generate_html(entries: list[dict], image_index: dict[str, str]) -> str:
    """Generate the review dashboard HTML."""

    # Enrich entries
    for entry in entries:
        entry["image_path"] = image_index.get(entry["fig_id"], "")
        current = find_current_text(entry["fig_id"], entry["qmd_file"])
        entry["current_cap"] = current["fig_cap"]
        entry["current_alt"] = current["fig_alt"]

    data_json = json.dumps(entries, indent=2, ensure_ascii=False)

    # The HTML template — image on top, diff below
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Figure Audit Review</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #f0f0f0; color: #333; max-width: 1100px; margin: 0 auto; padding: 16px;
}}
h1 {{ font-size: 22px; }}
.sub {{ color: #666; font-size: 13px; margin-bottom: 12px; }}
.mono {{ font-family: 'SF Mono','Fira Code','Consolas',monospace; }}

/* Stats bar */
.bar {{ display: flex; gap: 10px; margin: 10px 0; flex-wrap: wrap; align-items: center; }}
.pill {{ padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 700; }}
.pill-g {{ background: #d4edda; color: #1a5c2d; }}
.pill-y {{ background: #fff3cd; color: #7a5200; }}
.pill-r {{ background: #f9d6d5; color: #8b1a1a; }}
.pill-n {{ background: #e8e8e8; color: #555; }}

/* Filter buttons */
.filters button {{
  padding: 5px 12px; border: 1px solid #ccc; border-radius: 5px;
  background: #fff; cursor: pointer; font-size: 12px;
}}
.filters button:hover {{ background: #eee; }}
.filters button.on {{ background: #333; color: #fff; border-color: #333; }}
.kbd {{ font-family: monospace; font-size: 10px; background: #eee;
  padding: 1px 4px; border-radius: 2px; border: 1px solid #ccc; }}

/* ── Card ── */
.card {{
  background: #fff; border-radius: 10px; margin: 18px 0;
  box-shadow: 0 1px 8px rgba(0,0,0,0.07); overflow: hidden;
  border-left: 6px solid #ccc; transition: opacity 0.2s;
}}
.card[data-v="PASS"]       {{ border-left-color: #3d9e5a; }}
.card[data-v="MINOR"]      {{ border-left-color: #c87b2a; }}
.card[data-v="MISMATCH"]   {{ border-left-color: #c44; }}
.card.done {{ opacity: 0.35; }}
.card.focus {{ box-shadow: 0 0 0 3px #4a90c4; }}

/* Header row */
.ch {{ display: flex; justify-content: space-between; align-items: center;
  padding: 10px 16px; background: #fafafa; border-bottom: 1px solid #eee; }}
.ch h3 {{ font-size: 14px; }}
.ch .loc {{ font-size: 11px; color: #888; margin-left: 8px; }}

/* IMAGE — full width, big */
.fig {{
  background: #f7f7f7; display: flex; justify-content: center; align-items: center;
  padding: 20px; min-height: 200px; border-bottom: 1px solid #eee;
}}
.fig img {{ max-width: 100%; max-height: 520px; object-fit: contain; }}
.fig .na {{ color: #aaa; font-style: italic; }}

/* What Gemini sees */
.eye {{
  padding: 10px 16px; background: #f0f4ff; border-bottom: 1px solid #dde;
  font-size: 12px; line-height: 1.5; color: #555;
}}
.eye b {{ color: #333; }}

/* Two-column: current | fix */
.diff {{ display: grid; grid-template-columns: 1fr 1fr; }}
.diff .col {{ padding: 12px 16px; }}
.diff .col + .col {{ border-left: 1px solid #eee; }}
.lbl {{
  font-size: 9px; text-transform: uppercase; letter-spacing: 1.5px;
  color: #999; font-weight: 700; margin-bottom: 4px;
}}
.cur {{
  background: #fff5f5; border: 1px solid #fcc; border-radius: 5px;
  padding: 8px 10px; font-size: 12px; line-height: 1.55;
  white-space: pre-wrap; word-break: break-word;
}}
.fix {{
  background: #f0fff0; border: 1px solid #beb; border-radius: 5px;
  padding: 8px 10px; font-size: 12px; line-height: 1.55;
  white-space: pre-wrap; word-break: break-word;
}}
.ana {{
  background: #fafafa; border: 1px solid #eee; border-radius: 5px;
  padding: 8px 10px; font-size: 12px; line-height: 1.5; color: #555;
}}

/* Single-column sections */
.row {{ padding: 10px 16px; border-bottom: 1px solid #f0f0f0; }}

/* Actions */
.acts {{
  display: flex; gap: 8px; padding: 10px 16px; background: #fafafa;
  border-top: 1px solid #eee;
}}
.abtn {{
  padding: 7px 18px; border: 2px solid #3d9e5a; border-radius: 6px;
  background: #fff; cursor: pointer; font-weight: 700; font-size: 13px;
}}
.abtn:hover {{ background: #d4edda; }}
.abtn.ok {{ background: #d4edda; }}
.sbtn {{
  padding: 7px 18px; border: 1px solid #ccc; border-radius: 6px;
  background: #fff; cursor: pointer; font-size: 13px;
}}
.sbtn:hover {{ background: #f0f0f0; }}

/* Export bar */
.ebar {{
  position: sticky; bottom: 0; background: #fff; padding: 12px 16px;
  border-top: 2px solid #ddd; display: flex; justify-content: space-between;
  align-items: center; z-index: 10;
}}
.ebar .ct {{ font-weight: 700; }}
.ebtn {{
  padding: 8px 20px; background: #333; color: #fff; border: none;
  border-radius: 6px; font-size: 13px; font-weight: 700; cursor: pointer;
}}
.ebtn:hover {{ background: #555; }}
</style>
</head>
<body>

<h1>Figure-Narrative Audit Review</h1>
<p class="sub">Each card shows the rendered image + current text vs. suggested fix. Approve or skip.</p>

<div class="bar">
  <span class="pill pill-n" id="sT"></span>
  <span class="pill pill-g" id="sP"></span>
  <span class="pill pill-y" id="sM"></span>
  <span class="pill pill-r" id="sX"></span>
</div>
<div class="bar filters">
  <button class="on" onclick="flt('issues',this)">Issues Only</button>
  <button onclick="flt('all',this)">All</button>
  <button onclick="flt('mismatch',this)">Mismatches</button>
  <button onclick="flt('minor',this)">Minor</button>
  <button onclick="flt('pass',this)">Passes</button>
  <span style="color:#aaa;font-size:12px;margin-left:10px">
    <span class="kbd">j</span>/<span class="kbd">k</span> nav
    <span class="kbd">a</span> approve
    <span class="kbd">s</span> skip
  </span>
</div>

<div id="C"></div>

<div class="ebar">
  <div><span class="ct" id="nA">0</span> approved of <span id="nV">0</span></div>
  <div style="display:flex;gap:8px">
    <button class="ebtn" onclick="exp()">Export Approved (JSON)</button>
    <button class="sbtn" onclick="rst()">Reset</button>
  </div>
</div>

<script>
const D={data_json};
let F='issues', I=0;
const S=JSON.parse(localStorage.getItem('faR')||'{{}}');

function sv(){{ localStorage.setItem('faR',JSON.stringify(S)); }}
function K(e){{ return e.fig_id+':'+e.line; }}
function vc(v){{ return v==='PASS'?'PASS':v==='MINOR ISSUE'?'MINOR':'MISMATCH'; }}

function flt(f,btn){{
  F=f;
  document.querySelectorAll('.filters button').forEach(b=>b.classList.remove('on'));
  if(btn) btn.classList.add('on');
  draw();
}}

function vis(){{
  if(F==='all') return D;
  if(F==='issues') return D.filter(e=>e.verdict!=='PASS');
  if(F==='mismatch') return D.filter(e=>e.verdict==='MISMATCH');
  if(F==='minor') return D.filter(e=>e.verdict==='MINOR ISSUE');
  if(F==='pass') return D.filter(e=>e.verdict==='PASS');
  return D;
}}

function tog(k){{
  S[k]=S[k]==='yes'?undefined:'yes';
  if(!S[k]) delete S[k]; sv(); upd();
  const c=document.querySelector(`[data-k="${{k}}"]`);
  if(c){{ c.classList.toggle('done',S[k]==='yes');
    const b=c.querySelector('.abtn');
    if(b){{ b.textContent=S[k]==='yes'?'Approved':'Approve'; b.classList.toggle('ok',S[k]==='yes'); }}
  }}
}}

function nxt(){{ const v=vis(); I=Math.min(I+1,v.length-1); go(I); }}
function go(i){{
  const cs=document.querySelectorAll('.card');
  cs.forEach(c=>c.classList.remove('focus'));
  if(cs[i]){{ cs[i].classList.add('focus'); cs[i].scrollIntoView({{behavior:'smooth',block:'start'}}); I=i; }}
}}

function upd(){{
  const v=vis();
  document.getElementById('nA').textContent=v.filter(e=>S[K(e)]==='yes').length;
  document.getElementById('nV').textContent=v.length;
}}

function esc(s){{
  if(!s) return '';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n/g,'<br>');
}}

function draw(){{
  const v=vis();
  const nP=D.filter(e=>e.verdict==='PASS').length;
  const nM=D.filter(e=>e.verdict==='MINOR ISSUE').length;
  const nX=D.filter(e=>e.verdict==='MISMATCH').length;
  document.getElementById('sT').textContent=D.length+' total';
  document.getElementById('sP').textContent=nP+' pass';
  document.getElementById('sM').textContent=nM+' minor';
  document.getElementById('sX').textContent=nX+' mismatch';

  const box=document.getElementById('C'); box.innerHTML='';
  v.forEach((e,i)=>{{
    const k=K(e), ok=S[k]==='yes', vv=vc(e.verdict);
    const hasFix=e.suggested_fix && e.suggested_fix!=='No fix needed.';
    const img=e.image_path
      ? `<img src="${{e.image_path}}" alt="${{e.fig_id}}" loading="lazy">`
      : `<span class="na">No rendered image for ${{e.fig_id}}</span>`;

    // Parse suggested fix into separate caption and alt-text fixes
    const fixText = e.suggested_fix || '';
    // Try to extract separate caption and alt-text fixes from the suggestion
    let capFix = '', altFix = '';
    if (hasFix) {{
      // Look for caption fix patterns
      const capMatch = fixText.match(/(?:caption|fig-cap)[^:]*?:\\s*(?:```[^\\n]*\\n)?([\\s\\S]*?)(?:```|$)/i);
      const altMatch = fixText.match(/(?:alt-text|fig-alt)[^:]*?:\\s*(?:```[^\\n]*\\n)?([\\s\\S]*?)(?:```|$)/i);
      if (capMatch) capFix = capMatch[1].trim();
      if (altMatch) altFix = altMatch[1].trim();
      // If we couldn't parse separately, put the whole thing as the fix
      if (!capFix && !altFix) capFix = fixText;
    }}

    const okBadge = '<span style="color:#3d9e5a;font-size:18px;font-weight:700">&#10004; OK</span>';

    const diffHtml = `
      <div class="diff">
        <div class="col">
          <div class="lbl">Current Caption</div>
          <div class="cur">${{esc(e.current_cap||'(not extracted)')}}</div>
        </div>
        <div class="col">
          <div class="lbl">Suggested Caption</div>
          ${{capFix ? `<div class="fix">${{esc(capFix)}}</div>` : okBadge}}
        </div>
      </div>
      <div class="diff">
        <div class="col">
          <div class="lbl">Current Alt-Text</div>
          <div class="cur">${{esc(e.current_alt||'(not extracted)')}}</div>
        </div>
        <div class="col">
          <div class="lbl">Suggested Alt-Text</div>
          ${{altFix ? `<div class="fix">${{esc(altFix)}}</div>` : okBadge}}
        </div>
      </div>`;

    const card=document.createElement('div');
    card.className=`card ${{ok?'done':''}}`;
    card.dataset.v=vv;
    card.dataset.k=k;
    card.dataset.i=i;

    card.innerHTML=`
      <div class="ch">
        <div>
          <h3 class="mono">${{e.fig_id}}</h3>
          <span class="loc">${{e.chapter}} · line ${{e.line}}</span>
        </div>
        <span class="pill pill-${{vv==='PASS'?'g':vv==='MINOR'?'y':'r'}}">${{e.verdict}}</span>
      </div>
      <div class="fig">${{img}}</div>
      <div class="eye"><b>Gemini sees:</b> ${{esc(e.visual_description)}}</div>
      <div class="row">
        <div class="lbl">Caption Analysis</div>
        <div class="ana">${{esc(e.caption_analysis)}}</div>
      </div>
      <div class="row">
        <div class="lbl">Alt-Text Analysis</div>
        <div class="ana">${{esc(e.alttext_analysis)}}</div>
      </div>
      ${{diffHtml}}
      <div class="acts">
        <button class="abtn ${{ok?'ok':''}}" onclick="tog('${{k}}')">${{ok?'Approved':'Approve'}}</button>
        <button class="sbtn" onclick="nxt()">Skip</button>
      </div>`;
    box.appendChild(card);
  }});
  upd(); I=0;
}}

function exp(){{
  const ap=vis().filter(e=>S[K(e)]==='yes').map(e=>({{
    fig_id:e.fig_id, chapter:e.chapter, line:e.line, qmd_file:e.qmd_file,
    verdict:e.verdict, suggested_fix:e.suggested_fix,
    current_cap:e.current_cap, current_alt:e.current_alt,
  }}));
  const b=new Blob([JSON.stringify(ap,null,2)],{{type:'application/json'}});
  const u=URL.createObjectURL(b);
  const a=document.createElement('a'); a.href=u; a.download='approved_figure_fixes.json'; a.click();
  URL.revokeObjectURL(u);
}}

function rst(){{ if(confirm('Reset all?')){{ Object.keys(S).forEach(k=>delete S[k]); sv(); draw(); }} }}

document.addEventListener('keydown',(e)=>{{
  if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA') return;
  const v=vis();
  if(e.key==='j'||e.key==='ArrowDown'){{ e.preventDefault(); I=Math.min(I+1,v.length-1); go(I); }}
  if(e.key==='k'||e.key==='ArrowUp'){{ e.preventDefault(); I=Math.max(I-1,0); go(I); }}
  if(e.key==='a'){{ const x=v[I]; if(x) tog(K(x)); }}
  if(e.key==='s') nxt();
}});

draw();
</script>
</body>
</html>"""
    return html


def main():
    print("Building image index from rendered HTML...")
    image_index = build_image_index()
    print(f"  Found {len(image_index)} fig-id → image mappings\n")

    entries = parse_audit_files()
    if not entries:
        print("No audit data found. Run figure_audit_gemini.py first.")
        sys.exit(1)

    issues = [e for e in entries if e["verdict"] != "PASS"]
    passes = [e for e in entries if e["verdict"] == "PASS"]
    matched = sum(1 for e in entries if image_index.get(e["fig_id"]))
    print(f"Parsed {len(entries)} figure audits:")
    print(f"  PASS: {len(passes)}, Issues: {len(issues)}")
    print(f"  Image matched: {matched}/{len(entries)}")

    html = generate_html(entries, image_index)
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html, encoding="utf-8")

    print(f"\nDashboard: {OUTPUT_HTML}")
    print(f"\n  cd book/quarto && python3 -m http.server 8787")
    print(f"  http://localhost:8787/_build/figure_review.html")


if __name__ == "__main__":
    main()
