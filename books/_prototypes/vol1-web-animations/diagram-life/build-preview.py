"""Create an isolated review copy of the built chapter; never alter build output."""
from pathlib import Path
import os
import re
import json

here = Path(__file__).resolve().parent
books = here.parents[2]
catalog = json.loads((here.parent / 'animation-catalog/catalog.json').read_text())
source = books.parent / catalog['metadata']['build_root'] / 'vol1-html/output/vol1/10_model_compression/10_model_compression.html'
text = source.read_text()
base = os.path.relpath(source.parent, here) + '/'
text = text.replace('<head>', f'<head><base href="{base}">', 1)
assets = os.path.relpath(here, source.parent) + '/'
text = re.sub(r'src="[^"]*/quantized-matmul.js"', f'src="{assets}quantization-flow.js"', text)
text = re.sub(r'href="[^"]*/quantized-matmul.css"', f'href="{assets}quantization-flow.css"', text)
entry = """<script>
if(new URLSearchParams(location.search).has('explore')){
  const previewTimer=setInterval(()=>{
    const f=document.querySelector('[data-mlsys-explore-key="vol1:fig-weight-activations-quantization"]');
    if(f?.dataset.mlsysExploreState==='ready'){
      clearInterval(previewTimer);
      [...f.querySelectorAll('[role="tab"]')].find(t=>t.textContent.trim()==='Explore')?.click();
      f.style.scrollMarginTop='200px';f.scrollIntoView({block:'start'});
    }
  },100);
  setTimeout(()=>clearInterval(previewTimer),15000);
}
</script>"""
text = text.replace('</body>', entry + '</body>')
(here / 'quantization-flow.html').write_text(text)
print(here / 'quantization-flow.html')
