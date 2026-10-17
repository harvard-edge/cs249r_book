from pathlib import Path
import re,json,yaml
root=Path(__file__).resolve().parents[4];books=root/'books';out=Path(__file__).resolve().parent
def strings(x):
 if isinstance(x,str):yield x
 elif isinstance(x,list):
  for y in x:yield from strings(y)
 elif isinstance(x,dict):
  for y in x.values():yield from strings(y)
rows=[];excluded=[]
for v in range(1,5):
 cfg=yaml.safe_load((books/f'config/_quarto-pdf-vol{v}.yml').read_text())
 paths=list(dict.fromkeys(s for s in strings(cfg.get('book',{}).get('chapters',[])) if s.endswith('.qmd') and s.startswith(f'vol{v}/')))
 # Appendices may live in separate book.appendices rather than chapters.
 paths+= [s for s in strings(cfg.get('book',{}).get('appendices',[])) if s.endswith('.qmd') and s.startswith(f'vol{v}/') and s not in paths]
 about=f'vol{v}/frontmatter/about.qmd'
 if about not in paths and (books/about).exists():paths.insert(0,about) # Binder's generated index source for PDF/EPUB.
 for order,rel in enumerate(paths):
  p=books/rel
  if not p.exists():continue
  text=p.read_text();lines=text.splitlines();headings=[]
  for i,line in enumerate(lines):
   h=re.match(r'^(#{1,6})\s+(.+)',line)
   if h:headings.append((i,len(h[1]),h[2]))
   m=re.match(r'^(\s*:{3,})\s*\{#(fig-[\w-]+)\b',line)
   if not m:continue
   fence=m[1].strip();end=next((j for j in range(i+1,len(lines)) if lines[j].strip()==fence), min(i+200,len(lines)))
   block='\n'.join(lines[i:end+1]);attrs='\n'.join(lines[i:min(i+12,end+1)])
   def attr(name):
    found=re.search(r'\b'+name+r'="((?:[^"\\]|\\.)*)"',attrs,re.S)
    return found[1] if found else ''
   cap=attr('fig-cap');alt=attr('fig-alt')
   title=re.search(r'\*\*(.+?)\*\*',cap)
   section=next((h[2] for h in reversed(headings) if h[0]<i),'')
   assets=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',block)
   renderer='TikZ' if re.search(r'\{\.tikz\}|\\begin\{tikzpicture\}',block) else 'computed' if re.search(r'\{python\}|plt\.|mlsysim.viz',block) else 'SVG' if any('.svg' in a for a in assets) else 'raster' if assets else 'other'
   rows.append(dict(key=f'vol{v}:{m[2]}',volume=v,chapter_order=order,chapter=rel.split('/')[1],chapter_title=next((h[2] for h in headings if h[1]==1),rel.split('/')[1]),source='books/'+rel,line=i+1,figure_id=m[2],title=title[1] if title else m[2],caption=cap,alt=alt,section=section,renderer=renderer,assets=assets,context=re.sub(r'^\S*\s', '', '\n'.join(lines[max(0,i-4):i])[-2200:]) if len('\n'.join(lines[max(0,i-4):i]))>2200 else '\n'.join(lines[max(0,i-4):i])))
 canonical=set(paths)
 for p in (books/f'vol{v}').rglob('*.qmd'):
  if p.relative_to(books).as_posix() not in canonical:
   for i,line in enumerate(p.read_text().splitlines()):
    m=re.match(r'^\s*:{3,}\s*\{#(fig-[\w-]+)\b',line)
    if m:excluded.append(dict(volume=v,source=str(p.relative_to(root)),line=i+1,figure_id=m[1],reason='Source file absent from canonical PDF chapter/appendix list'))
(out/'inventory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
(out/'excluded-figures.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2))
for v in range(1,5):
 sub=[r for r in rows if r['volume']==v];(out/f'vol{v}-inventory.json').write_text(json.dumps(sub,ensure_ascii=False,indent=2));print(v,len(sub),len(set(r['chapter'] for r in sub)))
print('excluded',len(excluded),'missing captions',sum(not r['caption'] for r in rows))
