/* Original-artwork Explore mechanism. Shared player owns the pausable clock. */
(()=>{'use strict';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v1-dlrm-pipeline"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),NS='http://www.w3.org/2000/svg';
 const vb=svg.viewBox.baseVal,W=vb.width,H=vb.height;
 if(!W||!H)throw Error('Original artwork viewBox required');
 root.classList.add("original-v1-dlrm-pipeline");
 svg.setAttribute('role','img');svg.setAttribute('aria-label',"Why do dense MLPs and sparse embedding gathers stress opposite hardware limits?");
 svg.querySelectorAll('text').forEach(n=>{const f=n.getAttribute('fill');if(f)n.style.setProperty('fill',f,'important');});
 const overlay=document.createElementNS(NS,'g');overlay.setAttribute('data-explore-overlay','');svg.append(overlay);
 const mk=(tag,attrs={},text)=>{const el=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))el.setAttribute(k,String(v));if(text!=null)el.textContent=text;if(tag==='text'&&attrs.fill)el.style.setProperty('fill',String(attrs.fill),'important');return el;};
 const add=(tag,attrs={},text)=>{const el=mk(tag,attrs,text);overlay.append(el);return el;};
 add('path',{d:`M${(vb.x+W*.08).toFixed(1)},${(vb.y+H*.08).toFixed(1)}h${(W*.84).toFixed(1)}v${(H*.84).toFixed(1)}h-${(W*.84).toFixed(1)}z`,fill:'none',stroke:'none','data-mlsys-focus':''});
 const lerp=(a,b,t)=>a+(b-a)*t;
 const routePt=(pts,p)=>{if(pts.length===1)return pts[0];const lens=pts.slice(1).map((b,i)=>Math.hypot(b[0]-pts[i][0],b[1]-pts[i][1]));const tot=lens.reduce((s,x)=>s+x,0)||1;let rem=Math.max(0,Math.min(1,p))*tot,j=0;while(j<lens.length-1&&rem>lens[j]){rem-=lens[j];j++;}const q=lens[j]?rem/lens[j]:1;return[lerp(pts[j][0],pts[j+1][0],q),lerp(pts[j][1],pts[j+1][1],q)];};
 const canvas=document.createElement('div');canvas.className='mlsys-exp-canvas';canvas.tabIndex=0;canvas.setAttribute('aria-label','Original chapter diagram with synchronized state overlay.');canvas.append(svg);
 const readout=document.createElement('p');readout.className='mlsys-exp-readout';
 const invariant=document.createElement('p');invariant.className='mlsys-exp-invariant';
 const steps=[{"label": "Split dense and categorical inputs", "description": "Continuous dense features enter the left path toward the Bottom MLP while sparse categorical IDs enter the right path to index 3 multi-gigabyte embedding tables."}, {"label": "Transform dense features in Bottom MLP", "description": "Small weight matrices in the blue Bottom MLP project continuous inputs into one 128-dimensional dense activation vector with high arithmetic reuse in on-chip SRAM."}, {"label": "Gather 512-byte sparse rows", "description": "Each categorical ID performs a random HBM gather of one 128-dim FP32 row (512 bytes) with zero arithmetic reuse; a 1,000,000,000-user table spans 512.0 GB across multiple devices."}, {"label": "Align 4 shared-width vectors", "description": "Pooling/lookup yields 3 sparse 128-dim vectors matching the 1 dense 128-dim Bottom MLP vector—4 aligned representations ready for cross-feature correlation."}, {"label": "Compute 6 pairwise dot products", "description": "The purple Feature Interaction stage evaluates all C(4, 2) = 6 pairwise dot products among the 1 dense and 3 gathered sparse vectors to capture user-item affinities."}, {"label": "Predict click probability in Top MLP", "description": "Concatenating the 6 interaction scalars with the dense representation feeds the compute-dense Top MLP, emitting the final calibrated click-through probability."}];

 const regions=[
  [[9.5,191.7,60.2,18.4,'#006395'],[111.3,213.7,115.8,16.2,'#e67817']],
  [[9.5,143.1,60.2,23.2,'#006395']],
  [[121.8,133.8,20.8,60.2,'#e67817'],[158.8,133.8,20.8,60.2,'#e67817'],[195.8,133.8,20.8,60.2,'#e67817']],
  [[9.5,143.1,60.2,23.2,'#006395'],[121.8,112.9,20.8,9.3,'#e67817'],[158.8,112.9,20.8,9.3,'#e67817'],[195.8,112.9,20.8,9.3,'#e67817']],
  [[23.4,59.7,199.1,23.2,'#7e317b']],
  [[92.8,20.3,60.2,23.2,'#006395']]
 ];
 const hiLayer=add('g');
 const tokDense=add('circle',{r:4,fill:'#006395',stroke:'#fff','stroke-width':1.2,'data-motion-object':'true'});
 const tokSparse=[132.2,169.2,206.2].map(x=>add('circle',{cx:x,cy:221.8,r:3.8,fill:'#e67817',stroke:'#fff','stroke-width':1.2,'data-motion-object':'true'}));
 const tag=add('text',{x:228,y:28,'text-anchor':'middle','font-size':7.8,'font-weight':'700',fill:'#7e317b',stroke:'#fff','stroke-width':3.0,'paint-order':'stroke'},'');
 const densePts=[[39.6,200.9],[39.6,154.7],[39.6,154.7],[39.6,120],[85,71.3],[122.9,31.9]];
 const sparseY=[221.8,221.8,163.9,117.5,71.3,31.9];
 const readouts=[
  'Inputs split: continuous vector → Bottom MLP · 3 categorical IDs → sparse embedding tables',
  'Bottom MLP (compute-bound): dense GEMMs produce 1 × 128-dim FP32 vector (512 B)',
  'Sparse gather (memory-capacity & BW-bound): 3 × 512 B random reads; 10⁹ × 128 × 4 B = 512.0 GB/table',
  'Aligned width D = 128: 1 dense vector + 3 gathered sparse vectors = 4 vectors (2,048 B)',
  'Feature interaction: C(4, 2) = 6 pairwise dot products capture cross-feature correlations',
  'Top MLP → output: dense layers score concatenated features into click probability p(click)'
 ];
 function paint(from,to,p){
  const idx=p<1?from:to,moving=from!==to&&p<1;
  hiLayer.replaceChildren();
  regions[idx].forEach(([x,y,w,h,c])=>hiLayer.append(mk('rect',{x:x-1.5,y:y-1.5,width:w+3,height:h+3,rx:2.5,fill:c,'fill-opacity':'.12',stroke:c,'stroke-width':1.8})));
  const [dx,dy]=routePt([densePts[from],densePts[to]],p);
  tokDense.setAttribute('visibility',moving?'visible':'hidden');
  tokDense.setAttribute('cx',dx);tokDense.setAttribute('cy',dy);
  tokSparse.forEach((t,k)=>{
   t.setAttribute('visibility',moving?'visible':'hidden');
   const sx0=from>=5?122.9:from>=4?125+k*25:132.2+k*37;
   const sx1=to>=5?122.9:to>=4?125+k*25:132.2+k*37;
   t.setAttribute('cx',lerp(sx0,sx1,p));t.setAttribute('cy',lerp(sparseY[from],sparseY[to],p));
  });
  tag.textContent=['','dense GEMM (D=128)','3 × 512 B random gathers','4 vectors × 128-dim','C(4,2) = 6 dot products','p(click) emitted'][idx];
  readout.textContent=readouts[idx];
  invariant.textContent='Hardware split: Bottom/Top MLPs reuse weights in SRAM; 512.0 GB embedding tables demand HBM capacity and random-gather bandwidth.';
  root.dataset.values=JSON.stringify({step:idx,table_gb:512.0,row_bytes:512,pairwise_dots:6});
 }

 return window.MLSysOriginalPlayer.create({
  root,outer:context.outer,layout:'compact',
  title:"Why do dense MLPs and sparse embedding gathers stress opposite hardware limits?",
  intro:"Trace dense features and sparse 512-byte row gathers from 512.0 GB tables through 6 pairwise interactions into the top MLP.",
  scope:"mlsysim Scenarios.BillionUserTable: 1,000,000,000 rows \u00d7 128 FP32 values (4 bytes) = 512.0 GB per table (512 bytes per gathered row, zero arithmetic reuse), paired with 1 bottom-MLP dense vector and 3 sparse vectors producing C(4, 2) = 6 pairwise dot products.",
  steps,
  render(i,{stage,details}){if(!canvas.isConnected)stage.append(canvas);if(details&&!readout.isConnected)details.append(readout,invariant);root.dataset.sceneStep=String(i);paint(i,i,1);},
  transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>{root.dataset.animationProgress=String(p);paint(from,to,p);});}
 });
};})();
