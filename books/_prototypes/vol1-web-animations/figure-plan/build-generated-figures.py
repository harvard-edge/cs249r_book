#!/usr/bin/env python3
"""Build the author review gallery from verified PNGs and actual chapter HTML."""

import argparse
import hashlib
import html
import json
import os
import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def build():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', required=True, type=Path)
    args = parser.parse_args()
    run = args.build_root.resolve()
    manifest = json.loads((HERE / 'static-delivery.json').read_text())
    records = []
    for figure in manifest['figures']:
        proof = HERE / figure['png']
        assert hashlib.sha256(proof.read_bytes()).hexdigest() == figure['png_sha256']
        chapter = run / f"vol{figure['volume']}-html/output" / Path(figure['source']).relative_to('books').with_suffix('.html')
        soup = BeautifulSoup(chapter.read_text(), 'html.parser')
        node = soup.find(id=figure['figure_id'])
        assert node is not None, figure['key']
        lead = node.find_previous_sibling('p')
        payoff = node.find_next_sibling('p')
        caption = node.find('figcaption')
        assert lead is not None and payoff is not None and caption is not None
        record = {key: figure[key] for key in ('key', 'volume', 'title', 'source', 'renderer', 'png', 'png_sha256')}
        def plain_text(element):
            return re.sub(r'\s+', ' ', element.get_text()).strip()

        record.update(lead=plain_text(lead), caption=plain_text(caption), payoff=plain_text(payoff), chapter=os.path.relpath(chapter, HERE) + '#' + figure['figure_id'])
        record['review_fingerprint'] = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        records.append(record)
    escape = html.escape
    counts = Counter(record['volume'] for record in records)
    cards = []
    for index, record in enumerate(records, 1):
        key = escape(record['key'], quote=True)
        cards.append(f'''<article id="{key}" data-volume="{record['volume']}" data-key="{key}">
<div class="meta">{index} / {len(records)} · Volume {record['volume']} · {escape(record['renderer'])}</div>
<h2>{escape(record['title'])}</h2>
<p class="source">{escape(record['source'])}<br><code>{key}</code></p>
<div class="art"><a href="{escape(record['png'])}" target="_blank" rel="noopener"><img src="{escape(record['png'])}" alt="{escape(record['title'])}; full reviewed PNG" loading="lazy"></a></div>
<p><a href="{escape(record['png'])}" target="_blank" rel="noopener">Open full-resolution image</a> · <a href="{escape(record['chapter'])}" target="_blank" rel="noopener">Read in the rebuilt chapter</a></p>
<div class="passages"><h3>Chapter lead-in</h3><p>{escape(record['lead'])}</p><h3>Published caption</h3><p>{escape(record['caption'])}</p><h3>Following explanation</h3><p>{escape(record['payoff'])}</p></div>
<fieldset><legend>Your review</legend><label>Decision<select class="decision"><option value="pending">Not reviewed</option><option value="approved">Looks good</option><option value="changes">Changes needed</option></select></label><label>Notes<textarea class="notes" rows="3" placeholder="For example: simplify the caption or adjust this arrow."></textarea></label></fieldset>
</article>''')
    data = json.dumps(records, ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MLSysBook · New figure review</title><style>
:root{color-scheme:light;--ink:#24292f;--muted:#58616c;--line:#d8dfe6;--accent:#a71930;--surface:#f6f7f9}*{box-sizing:border-box}body{margin:0;background:white;color:var(--ink);font:16px/1.6 system-ui,-apple-system,sans-serif}header{border-top:5px solid var(--accent);border-bottom:1px solid var(--line)}main,.intro{max-width:1100px;margin:auto;padding:28px}h1{font-size:clamp(2rem,5vw,3rem);line-height:1.15}h2{font-size:1.55rem;line-height:1.3;margin:.4rem 0}h3{font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:1.4rem 0 .35rem}p{margin:.6rem 0 1rem}.meta,.source{color:var(--muted);font-size:.85rem}.source,code{overflow-wrap:anywhere}.notice{background:var(--surface);border-left:3px solid var(--accent);padding:14px 18px}.tools{display:flex;gap:16px;align-items:end;flex-wrap:wrap;margin-top:22px}label{display:grid;gap:6px;font-size:.9rem}select,button,textarea{font:inherit;color:inherit;background:white;border:1px solid #aab4be;border-radius:4px;padding:10px;min-height:44px}button{cursor:pointer}textarea{width:100%;resize:vertical}a{color:#0868bf}a:focus-visible,button:focus-visible,select:focus-visible,textarea:focus-visible{outline:3px solid #0868bf;outline-offset:3px}article{border:1px solid var(--line);border-top:4px solid var(--accent);border-radius:4px;padding:24px;margin-bottom:36px;scroll-margin-top:20px}.art{background:white;border:1px solid var(--line);padding:18px;margin:22px 0 12px}.art img{display:block;width:100%;height:auto}.passages{max-width:84ch}fieldset{border:1px solid var(--line);background:var(--surface);margin:24px 0 0;padding:16px;display:grid;grid-template-columns:200px 1fr;gap:18px}legend{font-weight:650;padding:0 8px}#storage{font-size:.85rem;color:var(--muted)}[hidden]{display:none!important}@media(max-width:600px){main,.intro{padding:18px}article{padding:14px}.art{padding:8px}fieldset{grid-template-columns:1fr}.tools{align-items:stretch;flex-direction:column}h2{font-size:1.3rem}}@media print{.tools,fieldset,#storage{display:none}article{break-before:page;border:0}.art{break-inside:avoid}}
</style></head><body><header><div class="intro"><strong>MLSysBook · Author review</strong><h1>New figures and their chapter explanations</h1>
<p>__COUNTS__ These are the new static additions, separate from the 98 selected interactive lessons and the existing figure inventory.</p>
<div class="notice">The images are the reviewed PNG proofs. The passages below are copied from the rebuilt chapter HTML without editorial rewriting. Each chapter link opens the figure in context. Technical and simulated learner reviews are complete within their recorded scope; author sign-off remains yours.</div>
<div class="tools"><label>Volume<select id="volume"><option value="">All volumes</option><option value="1">Volume I</option><option value="2">Volume II</option><option value="3">Volume III</option><option value="4">Volume IV</option></select></label><button id="export">Export my review</button><a href="index.html">Chapter audit and animation plans</a></div><p id="progress" aria-live="polite"></p><p id="storage">Decisions and notes stay in this browser. Export your review to save or share it; this page sends nothing.</p></div></header><main>__CARDS__</main>
<script type="application/json" id="records">__DATA__</script><script>
const records=JSON.parse(document.querySelector('#records').textContent),prefix='mlsysbook-static-author-review-v1:',rows=[...document.querySelectorAll('article')];
function read(r){try{return JSON.parse(localStorage.getItem(prefix+r.key+':'+r.review_fingerprint)||'{}')}catch{return {}}}
function update(){let approved=0,changes=0;for(const row of rows){approved+=row.querySelector('.decision').value==='approved';changes+=row.querySelector('.decision').value==='changes'}document.querySelector('#progress').textContent=`${approved} look good · ${changes} need changes · ${rows.length-approved-changes} not reviewed.`}
rows.forEach((row,i)=>{const record=records[i],saved=read(record),decision=row.querySelector('.decision'),notes=row.querySelector('.notes');decision.value=['approved','changes'].includes(saved.decision)?saved.decision:'pending';notes.value=saved.notes||'';function save(){try{localStorage.setItem(prefix+record.key+':'+record.review_fingerprint,JSON.stringify({decision:decision.value,notes:notes.value}))}catch{document.querySelector('#storage').textContent='Browser storage is unavailable. Export your review before closing this page.'}update()}decision.addEventListener('change',save);notes.addEventListener('input',save)});
document.querySelector('#volume').addEventListener('change',e=>{for(const row of rows)row.hidden=Boolean(e.target.value&&row.dataset.volume!==e.target.value)});
document.querySelector('#export').addEventListener('click',()=>{const review={exported_at:new Date().toISOString(),scope:'Author review of new static figures and integrated chapter passages',figures:records.map((r,i)=>({key:r.key,title:r.title,review_fingerprint:r.review_fingerprint,png_sha256:r.png_sha256,decision:rows[i].querySelector('.decision').value,notes:rows[i].querySelector('.notes').value}))};const url=URL.createObjectURL(new Blob([JSON.stringify(review,null,2)+'\\n'],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='mlsysbook-new-figure-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});update();
</script></body></html>'''
    count_text = f"{len(records)} figures across four volumes: " + ', '.join(f"{counts[v]} in Volume {v}" for v in sorted(counts)) + '.'
    page = page.replace('__COUNTS__', count_text).replace('__CARDS__', '\n'.join(cards)).replace('__DATA__', data)
    destination = HERE / 'generated-figures.html'
    destination.write_text(page)
    print(json.dumps({'output': os.path.relpath(destination, ROOT), 'figures': len(records), 'volumes': dict(counts)}, indent=2))


if __name__ == '__main__':
    build()
