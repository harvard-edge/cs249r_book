"""Embed a theme-inheriting tensor-layout exploration in generated chapter HTML.
Build first: ./binder/binder build html 07_frameworks --vol1 --skip-validate
Only generated HTML is changed. Run again safely from its saved static baseline.
"""
from pathlib import Path
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TARGET = ROOT / 'books/_build/html-vol1/chapters/07_frameworks/vol1/07_frameworks/07_frameworks.html'
if 'data-mlsys-explore=' in TARGET.read_text():
    raise SystemExit('Memory layout is now source-integrated. Build the chapter normally; do not apply the legacy injector.')
BASELINE = TARGET.with_name('07_frameworks.static.html')
if not BASELINE.exists() or 'id="memory-explore-script"' not in TARGET.read_text():
    BASELINE.write_bytes(TARGET.read_bytes())
chapter = BeautifulSoup(BASELINE.read_text(), 'html.parser')
outer = chapter.select_one('#fig-tensor-memory-layout')
assert outer is not None, 'Render the original frameworks chapter first.'
figure = outer if outer.name == 'figure' else outer.find('figure')
assert figure is not None
caption = figure.find('figcaption', recursive=False)
assert caption is not None
static = chapter.new_tag('div', id='memory-diagram')
for child in list(figure.contents):
    if child is not caption:
        static.append(child.extract())
figure.insert(0, static)
markup = '''<div id="memory-explore" hidden role="group" aria-label="Explore tensor memory layout">
<p class="memory-intro">Follow one tensor element into linear storage. Indices and offsets start at zero; strides below count <strong>elements, not bytes</strong>.</p>
<div class="memory-controls"><button type="button" data-action="play">▶ Play</button><button type="button" data-action="back">← Back</button><button type="button" data-action="next">Next →</button><button type="button" data-action="restart">Restart</button><label>Step <select data-role="step"></select></label><label>Layout <select data-role="layout"><option value="row">Row-major</option><option value="column">Column-major</option></select></label></div>
<div class="memory-panels"><section><h3>Tensor: 2 × 3</h3><p class="memory-small">Select any cell to inspect its address.</p><div class="memory-grid" aria-label="Tensor cells"></div></section><section><h3>Linear storage</h3><p class="memory-small">Element offsets from the storage origin</p><div class="memory-linear"></div></section></div>
<div class="memory-explanation" aria-live="polite" aria-atomic="true"></div><div class="memory-equation"></div>
<p class="memory-small">For an element size of B bytes, byte offset = element offset × B. Layout changes storage order, not the tensor values.</p></div>'''
player = BeautifulSoup(markup, 'html.parser').find('div')
figure.insert(1, player)
css = '''
#memory-explore{font:inherit;text-align:left;color:var(--bs-body-color);padding:.5rem 0}
#memory-explore *{box-sizing:border-box}
#memory-explore .memory-intro,#memory-explore .memory-small{font-size:.85rem;line-height:1.5;margin:.4rem 0 1rem}
#memory-explore .memory-controls{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap;margin:1rem 0}
#memory-explore button,#memory-explore select{font:inherit;font-size:.8rem;min-height:38px;padding:.35rem .55rem;border:1px solid var(--bs-border-color,#888);border-radius:4px;background:var(--bs-body-bg);color:var(--bs-body-color)}
#memory-explore button{cursor:pointer}#memory-explore button:disabled{opacity:.45;cursor:default}
#memory-explore label{font-size:.8rem;display:flex;gap:.35rem;align-items:center}
#memory-explore :focus-visible{outline:3px solid var(--bs-link-color,#a51c30);outline-offset:3px}
#memory-explore .memory-panels{display:grid;grid-template-columns:1fr 1.5fr;gap:1.4rem;margin:1rem 0}
#memory-explore h3{font-family:inherit;font-size:.9rem;font-weight:600;line-height:1.4;margin:.3rem 0;border:0;padding:0}
#memory-explore .memory-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
#memory-explore .memory-grid button{min-height:56px;border-radius:0;font-size:1rem}
#memory-explore .memory-linear{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:3px}
#memory-explore .memory-slot{text-align:center;min-width:0}
#memory-explore .memory-slot span{display:block;font-size:.72rem;color:var(--bs-secondary-color,#666);margin-bottom:5px}
#memory-explore .memory-value{border:1px solid var(--bs-border-color,#888);padding:.7rem .15rem;font-size:1rem}
#memory-explore .selected{background:color-mix(in srgb,var(--bs-link-color,#a51c30) 14%,var(--bs-body-bg))!important;outline:2px solid var(--bs-link-color,#a51c30);outline-offset:-2px;font-weight:700}
#memory-explore .memory-explanation{margin:1rem 0;font-size:.9rem;line-height:1.55}
#memory-explore .memory-equation{padding:.75rem;background:var(--bs-tertiary-bg,#f5f5f5);font-size:.9rem;overflow-wrap:anywhere}
@media(max-width:600px){#memory-explore .memory-panels{grid-template-columns:1fr;gap:1rem}#memory-explore .memory-grid{max-width:270px}#memory-explore label{width:100%}#memory-explore select{flex:1}}
@media print{#memory-explore{display:none!important}#memory-diagram{display:block!important}}
'''
for name, source in [('memory-explore-style', css), ('figure-views-style', (HERE/'figure-views.css').read_text())]:
    tag=chapter.new_tag('style',id=name);tag.string=source;chapter.head.append(tag)
shared=chapter.new_tag('script',id='figure-views-script');shared.string=(HERE/'figure-views.js').read_text();chapter.body.append(shared)
script = r'''
(()=>{const root=document.getElementById('memory-explore');const outer=document.getElementById('fig-tensor-memory-layout');const $=s=>root.querySelector(s);let stage=0,layout='row',chosen=4,running=false,timer=null;
const titles=['1. Select a cell','2. Read the strides','3. Compute the offset','4. Locate in storage','5. Compare layouts'];
titles.forEach((title,i)=>{const option=document.createElement('option');option.value=i;option.textContent=title;$('[data-role="step"]').append(option)});
for(let i=0;i<6;i++){const button=document.createElement('button');button.type='button';button.textContent=i+1;button.setAttribute('aria-label',`Value ${i+1}, row ${Math.floor(i/3)}, column ${i%3}`);button.onclick=()=>{pause();chosen=i;stage=2;render()};$('.memory-grid').append(button)}
for(let i=0;i<6;i++){const slot=document.createElement('div');slot.className='memory-slot';slot.innerHTML=`<span>offset ${i}</span><div class="memory-value"></div>`;$('.memory-linear').append(slot)}
function render(){const row=Math.floor(chosen/3),col=chosen%3,strides=layout==='row'?[3,1]:[1,2],offset=row*strides[0]+col*strides[1],order=layout==='row'?[1,2,3,4,5,6]:[1,4,2,5,3,6];
$('.memory-grid').querySelectorAll('button').forEach((b,i)=>{b.classList.toggle('selected',i===chosen);b.setAttribute('aria-pressed',String(i===chosen))});
$('.memory-linear').querySelectorAll('.memory-value').forEach((e,i)=>{e.textContent=order[i];e.classList.toggle('selected',stage>=3&&i===offset)});
const descriptions=[`Select value ${chosen+1} at row ${row}, column ${col}. The same tensor can use either storage layout.`,`${layout==='row'?'Row-major':'Column-major'} strides are [${strides}]. Moving down one row advances ${strides[0]} elements; moving right one column advances ${strides[1]} element${strides[1]===1?'':'s'}.`,`Multiply each zero-based index by its stride and add. This gives an element offset, not a byte address.`,`Value ${chosen+1} is stored at element offset ${offset}. The highlighted storage slot is the same element as the selected tensor cell.`,`Change the Layout menu to compare. Row-major stores neighboring columns consecutively; column-major stores neighboring rows consecutively. For this selected cell: row-major offset ${row*3+col}; column-major offset ${row+col*2}. Tensor coordinates and values stay the same.`];
$('.memory-explanation').textContent=descriptions[stage];$('.memory-equation').textContent=stage===0?`Selected tensor[${row}, ${col}] = ${chosen+1}`:stage===1?`Strides: [${strides}] elements`: `Element offset = ${row} × ${strides[0]} + ${col} × ${strides[1]} = ${offset}; byte offset = ${offset} × B`;
$('[data-role="step"]').value=stage;$('[data-role="layout"]').value=layout;$('[data-action="back"]').disabled=stage===0;$('[data-action="next"]').disabled=stage===4;$('[data-action="play"]').textContent=running?'Ⅱ Pause':stage===4?'↻ Replay':'▶ Play';}
function pause(){running=false;clearTimeout(timer);timer=null;render()}
function advance(){timer=setTimeout(()=>{stage++;if(stage===4){layout=layout==='row'?'column':'row';running=false;render();return}render();advance()},5500)}
$('[data-action="play"]').onclick=()=>{if(running){pause();return}if(stage===4){stage=0;chosen=4;layout='row'}running=true;render();advance()};
$('[data-action="back"]').onclick=()=>{pause();stage=Math.max(0,stage-1);render()};$('[data-action="next"]').onclick=()=>{pause();stage=Math.min(4,stage+1);render()};$('[data-action="restart"]').onclick=()=>{pause();stage=0;chosen=4;layout='row';render()};
$('[data-role="step"]').onchange=e=>{const requested=Number(e.target.value);pause();stage=requested;render()};$('[data-role="layout"]').onchange=e=>{const requested=e.target.value;pause();layout=requested;render()};
outer.addEventListener('mlsys-view-change',e=>{if(e.detail.view!=='explore')pause()});document.addEventListener('visibilitychange',()=>{if(document.hidden)pause()});new IntersectionObserver(entries=>{if(!entries[0].isIntersecting&&running)pause()},{threshold:.15}).observe(root);render();
window.MLSysFigureViews.enhance({figure:outer,diagram:document.getElementById('memory-diagram'),explore:root,initialView:'diagram'});
})();
'''
tag=chapter.new_tag('script',id='memory-explore-script');tag.string=script;chapter.body.append(tag)
TARGET.write_text(str(chapter))
print(TARGET)
