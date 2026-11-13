/* Original-artwork Explore mechanism. Shared player owns the pausable clock. */
(()=>{'use strict';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v1-attention-qkv"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),NS='http://www.w3.org/2000/svg';
 const vb=svg.viewBox.baseVal,W=vb.width,H=vb.height;
 if(!W||!H)throw Error('Original artwork viewBox required');
 root.classList.add("original-v1-attention-qkv");
 svg.setAttribute('role','img');svg.setAttribute('aria-label',"How do queries, keys, and values form the 6\u00d76 attention matrix and outputs?");
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
 const steps=[{"label": "Project 6 tokens into Q, K, and V", "description": "Three learned projection matrices map the 6 input tokens ('Data', 'visualization', 'em', 'powers', 'users', 'to') into cyan Query rows on the left, red Key columns above, and green Value vectors below."}, {"label": "Match one query row against 6 keys", "description": "Selecting token 4 ('powers') sweeps its cyan Query vector across all 6 red Key columns to evaluate 6 dot-product similarity scores along row 4."}, {"label": "Populate all 36 pairwise cells", "description": "Repeating query-key matching across all 6 rows materializes the full 6 × 6 = 36 similarity matrix—exposing the O(N²) score footprint that naive attention stores in memory."}, {"label": "Normalize row affinities", "description": "Row-wise softmax scales and normalizes row 4 ('powers') into non-negative attention weights [0.06, 0.11, 0.34, 0.25, 0.16, 0.08] that sum to 1.00, peaking on sibling subword 'em' and itself."}, {"label": "Aggregate weighted green values", "description": "Row 4's normalized weights scale all 6 green Value vectors from below and sum them into contextualized Output row 4 on the right."}, {"label": "Emit all 6 contextualized outputs", "description": "Applying each row's normalized weights across the shared green Value bank produces all 6 contextualized output representations while earlier Keys and Values remain reusable."}];

 const xs=[154.2,164.8,175.4,186.1,196.7,207.3],ys=[78.9,89.5,100.1,110.8,121.4,132.0];
 const qBox=add('rect',{x:68,y:73,width:74,height:64,rx:3,fill:'#006395','fill-opacity':'.08',stroke:'#006395','stroke-width':1.5,visibility:'hidden'});
 const kBox=add('rect',{x:147,y:10,width:66,height:61,rx:3,fill:'#b91c1c','fill-opacity':'.08',stroke:'#b91c1c','stroke-width':1.5,visibility:'hidden'});
 const vBox=add('rect',{x:147,y:142,width:66,height:56,rx:3,fill:'#008f45','fill-opacity':'.08',stroke:'#008f45','stroke-width':1.5,visibility:'hidden'});
 const oBox=add('rect',{x:218,y:73,width:106,height:64,rx:3,fill:'#7e317b','fill-opacity':'.08',stroke:'#7e317b','stroke-width':1.5,visibility:'hidden'});
 const gridBox=add('rect',{x:148,y:73,width:65.5,height:65.5,rx:3,fill:'none',stroke:'#006395','stroke-width':1.8,visibility:'hidden'});
 const rowHi=add('rect',{x:148.5,y:ys[3]-5.6,width:64.5,height:11.2,rx:2.5,fill:'#006395','fill-opacity':'.16',stroke:'#006395','stroke-width':1.6,visibility:'hidden'});
 const outRowHi=add('rect',{x:219,y:ys[3]-5.6,width:104,height:11.2,rx:2.5,fill:'#008f45','fill-opacity':'.18',stroke:'#008f45','stroke-width':1.6,visibility:'hidden'});
 const probe=add('circle',{r:5.2,fill:'none',stroke:'#e67817','stroke-width':2,'data-motion-object':'true',visibility:'hidden'});
 const callout=add('text',{x:180.8,y:69.5,'text-anchor':'middle','font-size':7.8,'font-weight':'700',fill:'#006395',stroke:'#fff','stroke-width':2.8,'paint-order':'stroke'},'');
 const weights=[0.06,0.11,0.34,0.25,0.16,0.08];
 const readouts=[
  'Sequence N = 6 tokens · 3 projections (W_Q, W_K, W_V) cost 3 d² parameters per layer',
  'Row 4 ("powers") dot-product scan: q₄ · [k₁..k₆] → 6 similarity evaluations',
  'Full score matrix QKᵀ: 6 × 6 = 36 cells materialized (O(N²) intermediate state)',
  'Softmax(row 4): α₄ = [0.06, 0.11, 0.34, 0.25, 0.16, 0.08] · Σ_j α_{4,j} = 1.00',
  'Value aggregation: o₄ = Σ_{j=1..6} α_{4,j} v_j → contextualized vector for "powers"',
  'All 6 output vectors emitted from 36 attention weights × 6 green Value vectors'
 ];
 function paint(from,to,p){
  const idx=p<1?from:to;
  qBox.setAttribute('visibility',idx<=1?'visible':'hidden');
  kBox.setAttribute('visibility',idx<=1?'visible':'hidden');
  vBox.setAttribute('visibility',(idx===0||idx>=4)?'visible':'hidden');
  oBox.setAttribute('visibility',idx===5?'visible':'hidden');
  gridBox.setAttribute('visibility',(idx===2||idx===5)?'visible':'hidden');
  rowHi.setAttribute('visibility',(idx===1||idx===3||idx===4)?'visible':'hidden');
  outRowHi.setAttribute('visibility',idx===4?'visible':'hidden');
  probe.setAttribute('visibility','hidden');
  if((from===0&&to===1)||idx===1){
   probe.setAttribute('visibility','visible');
   const q=from===0&&to===1?p:1,col=Math.min(5,Math.floor(q*5.99));
   probe.setAttribute('cx',lerp(xs[0],xs[5],q));probe.setAttribute('cy',ys[3]);
   callout.textContent=`q₄ · k_${col+1} (${col+1}/6)`;
  }else if(from===3&&to===4||idx===4){
   probe.setAttribute('visibility','visible');
   const [x,y]=routePt([[180.8,ys[3]],[180.8,155],[255,ys[3]]],from===3&&to===4?p:1);
   probe.setAttribute('cx',x);probe.setAttribute('cy',y);
   callout.textContent='o₄ = Σ α₄ⱼ vⱼ';
  }else{
   callout.textContent=['Q, K, V ready (N=6)','','6 × 6 = 36 score cells','Σ α₄ⱼ = 1.00','','6 outputs complete'][idx];
  }
  readout.textContent=readouts[idx];
  invariant.textContent='Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V · N = 6 tokens → N² = 36 pairwise weights.';
  root.dataset.values=JSON.stringify({step:idx,seq_len:6,cells:36,row4_weights:weights});
 }

 return window.MLSysOriginalPlayer.create({
  root,outer:context.outer,layout:'compact',
  title:"How do queries, keys, and values form the 6\u00d76 attention matrix and outputs?",
  intro:"Follow a 6-token sequence through query-key matching across all 36 score cells and value aggregation into contextualized outputs.",
  scope:"6-token sequence ('Data', 'visualization', 'em', 'powers', 'users', 'to') with 3 projection matrices (W_Q, W_K, W_V costing 3 d\u00b2 parameters) and N\u00b2 = 36 pairwise attention cells. Row weights are illustrative normalized affinities.",
  steps,
  render(i,{stage,details}){if(!canvas.isConnected)stage.append(canvas);if(details&&!readout.isConnected)details.append(readout,invariant);root.dataset.sceneStep=String(i);paint(i,i,1);},
  transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>{root.dataset.animationProgress=String(p);paint(from,to,p);});}
 });
};})();
