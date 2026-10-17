"""Embed the audited convolution player in a locally rendered original chapter.
Run binder build html 06_nn_architectures --vol1 --skip-validate first.
Only generated HTML is modified; original chapter source is untouched.
"""
from pathlib import Path
import re
from bs4 import BeautifulSoup
import tinycss2

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BUILD=ROOT/'books/_build/html-vol1/chapters/06_nn_architectures'
TARGET=BUILD/'vol1/06_nn_architectures/06_nn_architectures.html'
if 'data-mlsys-explore=' in TARGET.read_text():
    raise SystemExit('This chapter now uses source-integrated figures. Build normally; do not apply the legacy injector.')
prototype=BeautifulSoup((HERE/'convolution-in-context.html').read_text(),'html.parser')
backup=TARGET.with_name('06_nn_architectures.static.html')
if not backup.exists() or 'id="cnn-player-script"' not in TARGET.read_text():
    backup.write_bytes(TARGET.read_bytes())
chapter=BeautifulSoup(backup.read_text(),'html.parser')
original=chapter.select_one('#fig-cnn figure')
assert original is not None,'Build the original chapter first'
caption=original.find('figcaption',recursive=False)
assert caption is not None
player=chapter.new_tag('div',id='cnn-player')
player['class']=['mlsys-inline-animation']
player['role']='group'
player['aria-label']='Animated convolution example'
source=prototype.find('figure')
for child in list(source.contents):
    if getattr(child,'name',None)=='figcaption': continue
    player.append(child.extract())
player.select_one('.figure-title').decompose()
player.select_one('.setup').string='Illustrative single-channel calculation · stride 1 · no padding · zero bias · no activation'
ids={node['id']:'cnn-'+node['id'] for node in player.select('[id]')}
for node in player.select('[id]'): node['id']=ids[node['id']]
for node in player.select('[for]'): node['for']=ids.get(node['for'],node['for'])
# Original artwork and caption remain available, and provide the print/no-JS form.
fallback=chapter.new_tag('div')
fallback['class']=['cnn-static-fallback']
for child in list(original.contents):
    if child is caption: continue
    fallback.append(child.extract())
original.insert(0,fallback)
original.insert(0,player)
# No activation is hidden unless initialization succeeds.
player['hidden']=''
css=prototype.style.string
excluded={'*',':root','body','header','header b','main','h1','p','figure','figcaption','.note','.note a','.related','.related a'}
def scope(rules):
    result=[]
    for rule in rules:
        if rule.type=='qualified-rule':
            selectors=tinycss2.serialize(rule.prelude).strip().split(',')
            selectors=[s.strip() for s in selectors if s.strip() not in excluded]
            if selectors:
                result.append(','.join('#cnn-player '+s for s in selectors)+'{'+tinycss2.serialize(rule.content)+'}')
        elif rule.type=='at-rule' and rule.content and rule.lower_at_keyword=='media':
            result.append('@media '+tinycss2.serialize(rule.prelude)+'{'+scope(tinycss2.parse_rule_list(rule.content))+'}')
    return ''.join(result)
scoped=scope(tinycss2.parse_stylesheet(css))
overrides='''
#cnn-player{font-family:inherit;font-size:.875rem;line-height:1.5;color:var(--bs-body-color);text-align:left;width:100%;--accent:var(--bs-link-color,#a51c30);--muted:var(--bs-secondary-color,#626262);--line:var(--bs-border-color,#bbb);--orange:#e67817;--blue:#006395}
#cnn-player *{box-sizing:border-box}
#cnn-player .controls{position:static;background:transparent;margin:0 0 1rem;border-color:var(--bs-border-color)}
#cnn-player button,#cnn-player select{font-family:inherit;background:var(--bs-body-bg);color:var(--bs-body-color);border-color:var(--bs-border-color)}
#cnn-player button.primary{background:var(--bs-link-color,#a51c30);color:var(--bs-body-bg);border-color:var(--bs-link-color,#a51c30)}
#cnn-player button:hover{background:var(--bs-tertiary-bg)}
#cnn-player button.primary:hover{filter:brightness(.9)}
#cnn-player .matrix{background:var(--bs-body-bg);border-color:var(--bs-border-color)}
#cnn-player .cell{border-color:var(--bs-border-color);color:var(--bs-body-color)}
#cnn-player .matrix.input{border:1px dashed var(--bs-border-color)}
#cnn-player .input .cell:nth-child(7n){border-right:0}
#cnn-player .input .cell:nth-child(n+43){border-bottom:0}
#cnn-player .input .cell{color:var(--bs-body-color)}
#cnn-player .input .bright{background:color-mix(in srgb,var(--bs-body-color) 25%,var(--bs-body-bg))}
#cnn-player .filter .cell{background:color-mix(in srgb,#e67817 18%,var(--bs-body-bg))}
#cnn-player .output .cell{color:var(--bs-body-color);background:color-mix(in srgb,#7e317b 12%,var(--bs-body-bg))}
#cnn-player .output .unknown{color:var(--bs-secondary-color);background:var(--bs-body-bg)}
#cnn-player .destination{border-color:#7e317b}
#cnn-player .matched{outline-color:#9e65a5;background:color-mix(in srgb,var(--bs-link-color) 18%,var(--bs-body-bg))!important}
#cnn-player .pair-term{background:color-mix(in srgb,var(--bs-link-color) 18%,var(--bs-body-bg));color:var(--bs-body-color)}
#cnn-player .panel h2{border:0;padding:0;font-family:inherit;color:var(--bs-body-color);letter-spacing:normal}
#cnn-player .panel h2 span,#cnn-player .position,#cnn-player .setup,#cnn-player .math-label,#cnn-player .row-label{color:var(--bs-secondary-color)}
#cnn-player .math,#cnn-player .mobile-cue{background:var(--bs-tertiary-bg);color:var(--bs-body-color)}
#cnn-player .math-result{font-family:inherit}
#cnn-player .unchanged{color:var(--bs-body-color)}
#cnn-player .diagram{margin:1rem auto}
#cnn-player .explanation{min-height:0;margin:1rem 0}

@media print{#fig-cnn #cnn-player{display:none!important}#fig-cnn.cnn-enhanced .cnn-static-fallback{display:block}}
'''
style=chapter.new_tag('style',id='cnn-player-style')
style.string=scoped+overrides+(HERE/'figure-views.css').read_text()
chapter.head.append(style)
script=prototype.find('script').string
script=script.replace("const $=s=>document.querySelector(s)","const $=s=>root.querySelector(s)")
script=script.replace('blue-outlined','violet-outlined').replace('blue destination','violet destination').replace('one blue-outlined','one violet-outlined')
script=script.replace('function pause(){running=false;clear();render(false)}', 'function pause(){running=false;clear();render(false);root.getAnimations({subtree:true}).forEach(a=>{try{a.finish()}catch{a.cancel()}})}')
script=script.replace('document.querySelectorAll(', 'root.querySelectorAll(')
script=script.replace(".observe($('figure'))",'.observe(root)')
for old,new in sorted(ids.items(),key=lambda kv:-len(kv[0])):
    script=re.sub(r'#'+re.escape(old)+r"(?=['\"\s])", '#'+new,script)
script="(()=>{const root=document.getElementById('cnn-player');try{\n"+script+"\nconst figure=root.closest('figure');figure.addEventListener('mlsys-view-change',()=>pause());window.MLSysFigureViews.enhance({figure,diagram:figure.querySelector('.cnn-static-fallback'),explore:root});}catch(error){console.error('Convolution player:',error)}})();"
tag=chapter.new_tag('script',id='cnn-player-script');tag.string=(HERE/'figure-views.js').read_text()+'\n'+script
chapter.body.append(tag)
TARGET.write_text(str(chapter))
print(TARGET)
