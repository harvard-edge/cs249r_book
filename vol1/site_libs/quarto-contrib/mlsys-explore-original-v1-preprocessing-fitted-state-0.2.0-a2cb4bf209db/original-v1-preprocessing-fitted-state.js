/* Original-artwork Explore mechanism. Shared player owns the pausable clock. */
(()=>{'use strict';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v1-preprocessing-fitted-state"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),NS='http://www.w3.org/2000/svg';
 const vb=svg.viewBox.baseVal,W=vb.width,H=vb.height;
 if(!W||!H)throw Error('Original artwork viewBox required');
 root.classList.add("original-v1-preprocessing-fitted-state");
 svg.setAttribute('role','img');svg.setAttribute('aria-label',"Why must validation and serving reuse training-fitted preprocessing state?");
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
 const steps=[{"label": "Isolate the training split", "description": "Only training records x = [10, 20, 30] enter the fitting boundary; held-out validation, test, and live serving inputs stay excluded so future statistics never leak into learned parameters."}, {"label": "Fit statistics once", "description": "The Fit operation computes μ = 20.0 and σ = 8.165 once on the training split and writes those frozen parameters alongside the vocabulary."}, {"label": "Seal the shared artifact", "description": "Versioned transformation code and frozen state (μ = 20.0, σ = 8.165) are packaged into a single immutable release artifact carried with the model weights."}, {"label": "Fan out to 3 apply paths", "description": "All 3 downstream paths—training records, validation/test splits, and live serving requests—load the exact same frozen artifact without re-estimating statistics."}, {"label": "Verify raw-record parity", "description": "Feeding raw query x = 50.0 through the shared artifact produces z = (50.0 − 20.0) / 8.165 = +3.67 on every path, passing the parity check with 0.00 σ skew."}, {"label": "Contrast independent refit skew", "description": "Refitting on a shifted batch [50, 70, 90] (μ = 70.0, σ = 16.33) maps the exact same x = 50.0 to z = -1.22 instead of +3.67—a 4.90 σ training-serving skew that silently corrupts predictions."}];

 const boxes=[[4,0.4,89,31.1,'#008f45'],[111.4,0.4,77.7,31.1,'#e67817'],[232.8,0.4,106,31.1,'#006395'],[4,48.4,337.6,36.7,'#006395'],[1.2,102.1,94.7,31.1,'#e67817'],[125.5,102.1,94.7,31.1,'#e67817'],[249.8,102.1,94.7,31.1,'#e67817'],[2,146,342,49,'#008f45']];
 const hi=boxes.map(([x,y,w,h,c])=>add('rect',{x:x-1.5,y:y-1.5,width:w+3,height:h+3,rx:3,fill:c,'fill-opacity':'.10',stroke:c,'stroke-width':1.8,visibility:'hidden'}));
 const tokA=add('circle',{r:4.2,fill:'#006395',stroke:'#fff','stroke-width':1.3,'data-motion-object':'true'});
 const tokB=add('circle',{r:4.2,fill:'#006395',stroke:'#fff','stroke-width':1.3,'data-motion-object':'true',visibility:'hidden'});
 const tokC=add('circle',{r:4.2,fill:'#006395',stroke:'#fff','stroke-width':1.3,'data-motion-object':'true',visibility:'hidden'});
 const badge=add('text',{x:173,y:44.5,'text-anchor':'middle','font-size':8.0,'font-weight':'700',fill:'#006395',stroke:'#fff','stroke-width':3.2,'paint-order':'stroke'},'');
 const readouts=[
  'Split boundary: train [10, 20, 30] admitted to Fit · validation/test/serving excluded from Fit',
  'Fitted once on train: μ_train = 20.0, σ_train = 8.165 → frozen state locked',
  'Shared artifact v1: code + frozen (μ = 20.0, σ = 8.165) bundled with model weights',
  '3 apply paths loaded: train, validation/test, and serving execute identical frozen transform',
  'Parity PASS on raw x = 50.0: z_train = +3.67 · z_val = +3.67 · z_serv = +3.67 (skew = 0.00 σ)',
  'Refit anti-pattern on [50, 70, 90]: μ_refit = 70.0, σ_refit = 16.33 → z_refit = -1.22 vs +3.67 (4.90 σ skew!)'
 ];
 function paint(from,to,p){
  const idx=p<1?from:to,moving=from!==to&&p<1;
  const active=[[0],[1,2],[2,3],[3,4,5,6],[4,5,6,7],[6,7]][idx];
  hi.forEach((el,k)=>{const on=active.includes(k);el.setAttribute('visibility',on?'visible':'hidden');if(idx===5&&(k===6||k===7)){el.setAttribute('stroke','#b91c1c');el.setAttribute('fill','#b91c1c');}else{el.setAttribute('stroke',boxes[k][4]);el.setAttribute('fill',boxes[k][4]);}});
  tokA.setAttribute('visibility',moving?'visible':'hidden');
  tokB.setAttribute('visibility','hidden');tokC.setAttribute('visibility','hidden');
  tokA.setAttribute('fill',idx===5?'#b91c1c':idx===0?'#008f45':'#006395');
  if(from===0&&to===1){const [x,y]=routePt([[48.5,16],[150.2,16],[285.8,16]],p);tokA.setAttribute('cx',x);tokA.setAttribute('cy',y);}
  else if(from===1&&to===2){const [x,y]=routePt([[285.8,16],[285.8,66.8],[172.8,66.8]],p);tokA.setAttribute('cx',x);tokA.setAttribute('cy',y);}
  else if(from===2&&to===3){
   tokB.setAttribute('visibility',moving?'visible':'hidden');tokC.setAttribute('visibility',moving?'visible':'hidden');
   const p1=routePt([[48.5,85],[48.5,117.6]],p),p2=routePt([[172.8,85],[172.8,117.6]],p),p3=routePt([[297.1,85],[297.1,117.6]],p);
   tokA.setAttribute('cx',p1[0]);tokA.setAttribute('cy',p1[1]);tokB.setAttribute('cx',p2[0]);tokB.setAttribute('cy',p2[1]);tokC.setAttribute('cx',p3[0]);tokC.setAttribute('cy',p3[1]);
  }else{
   const anchors=[[48.5,16],[285.8,16],[172.8,66.8],[172.8,117.6],[172.8,164],[297.1,164]];
   const [x,y]=routePt([anchors[from],anchors[to]],p);tokA.setAttribute('cx',x);tokA.setAttribute('cy',y);
  }
  badge.setAttribute('fill',idx===5?'#b91c1c':idx>=4?'#008f45':'#006395');
  badge.textContent=['fit input: train only','frozen: μ=20.0, σ=8.165','artifact sealed with weights','identical state → 3 paths','x=50 → z=+3.67 (parity OK)','refit skew: z=-1.22 (Δ=4.90σ)'][idx];
  readout.textContent=readouts[idx];
  invariant.textContent='Rule: fit(D_train) → θ_prep once; apply transform(x; θ_prep) unchanged across train, val/test, and serving.';
  root.dataset.values=JSON.stringify({step:idx,z_shared:3.67,z_refit:-1.22,skew_sigma:idx===5?4.9:0});
 }

 return window.MLSysOriginalPlayer.create({
  root,outer:context.outer,layout:'compact',
  title:"Why must validation and serving reuse training-fitted preprocessing state?",
  intro:"Trace how fitting preprocessing once on the training split freezes shared parameters across 3 application paths and prevents a 4.90 \u03c3 serving skew.",
  scope:"Illustrative z-score normalization toy computed in new-explore-batch.calc.py: training split [10, 20, 30] yields \u03bc = 20.0 and \u03c3 = 8.165, whereas an independently refitted serving batch [50, 70, 90] yields \u03bc = 70.0 and \u03c3 = 16.330, turning raw input x = 50.0 from z = +3.67 into z = -1.22 (4.90 \u03c3 skew).",
  steps,
  render(i,{stage,details}){if(!canvas.isConnected)stage.append(canvas);if(details&&!readout.isConnected)details.append(readout,invariant);root.dataset.sceneStep=String(i);paint(i,i,1);},
  transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>{root.dataset.animationProgress=String(p);paint(from,to,p);});}
 });
};})();
