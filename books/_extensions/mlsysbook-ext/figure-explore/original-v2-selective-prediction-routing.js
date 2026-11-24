/* Original-artwork Explore mechanism. Shared player owns the pausable clock. */
(()=>{'use strict';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v2-selective-prediction-routing"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),NS='http://www.w3.org/2000/svg';
 const vb=svg.viewBox.baseVal,W=vb.width,H=vb.height;
 if(!W||!H)throw Error('Original artwork viewBox required');
 root.classList.add("original-v2-selective-prediction-routing");
 svg.setAttribute('role','img');svg.setAttribute('aria-label',"How does raising threshold \u03c4 trade accepted risk against fallback load?");
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
 const steps=[{"label": "Score an incoming request", "description": "An incoming request x (at 1,000 req/s arrival rate) enters the primary Model, which emits prediction f(x) alongside a calibrated confidence and OOD competence score κ(x)."}, {"label": "Accept high-competence traffic at τ = 0.55", "description": "With threshold τ = 0.55, a request scoring κ(x) = 0.91 ≥ τ takes the green Answer path; 90% coverage leaves 0.082 selective risk and 100 req/s fallback load."}, {"label": "Escalate borderline traffic to Route", "description": "Raising τ to 0.72 lowers selective risk to 0.041 at 78% coverage; a borderline request with κ(x) = 0.66 < 0.72 diverts to the violet Route path (stronger model or human reviewer), growing fallback load to 220 req/s."}, {"label": "Abstain on out-of-distribution inputs", "description": "An out-of-distribution request scoring κ(x) = 0.18 falls far below τ = 0.72 and routes to the red Abstain path, returning a validated safe default within the latency budget."}, {"label": "Raise τ to 0.85 and budget fallback capacity", "description": "Tightening τ to 0.85 slides the operating point down the risk-coverage curve to 0.019 risk and 62% coverage—sending 38% of traffic (380 req/s) into Route and Abstain queues that must absorb the load."}];

 const boxDefs=[[20,105,96,56,'#64748b'],[146,88,138,90,'#1F407A'],[314,93,112,80,'#7C3AED'],[472,30,200,52,'#16A34A'],[472,107,200,52,'#7C3AED'],[472,184,200,52,'#DC2626']];
 const hi=boxDefs.map(([x,y,w,h,c])=>add('rect',{x:x-2,y:y-2,width:w+4,height:h+4,rx:4,fill:c,'fill-opacity':'.10',stroke:c,'stroke-width':2,visibility:'hidden'}));
 const pkt=add('circle',{r:6,fill:'#1F407A',stroke:'#fff','stroke-width':1.8,'data-motion-object':'true'});
 const tauDot=add('circle',{cx:245,cy:267,r:6,fill:'#7C3AED',stroke:'#fff','stroke-width':2,'data-motion-object':'true'});
 const badge=add('text',{x:572,y:314,'text-anchor':'middle','font-size':10.5,'font-weight':'700',fill:'#7C3AED',stroke:'#fff','stroke-width':3.2,'paint-order':'stroke'},'');
 const curvePts=[[245,267],[245,267],[190,286],[190,286],[130,292.5]];
 const flowRoutes=[
  [[68,133],[215,133]],
  [[215,133],[370,133],[446,120],[446,56],[572,56]],
  [[215,133],[370,133],[572,133]],
  [[215,133],[370,133],[446,146],[446,210],[572,210]],
  [[370,133],[572,133]]
 ];
 const activeBoxes=[[0,1],[2,3],[2,4],[2,5],[2,4,5]];
 const colors=['#1F407A','#16A34A','#7C3AED','#DC2626','#7C3AED'];
 const readouts=[
  'Arrival rate λ = 1,000 req/s · Model emits prediction f(x) + competence score κ(x)',
  'τ = 0.55 · κ(x) = 0.91 ≥ τ → Answer · coverage = 90%, selective risk = 0.082, fallback = 100 req/s',
  'τ = 0.72 · κ(x) = 0.66 < τ → Route (reviewer/stronger model) · coverage = 78%, risk = 0.041, fallback = 220 req/s',
  'τ = 0.72 · κ(x) = 0.18 ≪ τ → Abstain (validated safe default) · OOD hazard contained',
  'Raise τ → 0.85 · coverage = 62%, selective risk = 0.019 · fallback load = (1 − 0.62) × 1,000 = 380 req/s'
 ];
 function paint(from,to,p){
  const idx=p<1?from:to,moving=from!==to&&p<1;
  hi.forEach((el,k)=>el.setAttribute('visibility',activeBoxes[idx].includes(k)?'visible':'hidden'));
  const r=from!==to?flowRoutes[to]:flowRoutes[idx],q=from!==to?p:1,[x,y]=routePt(r,q);
  pkt.setAttribute('visibility',moving?'visible':'hidden');
  pkt.setAttribute('cx',x);pkt.setAttribute('cy',y);pkt.setAttribute('fill',colors[idx]);
  tauDot.setAttribute('cx',lerp(curvePts[from][0],curvePts[to][0],p));
  tauDot.setAttribute('cy',lerp(curvePts[from][1],curvePts[to][1],p));
  badge.setAttribute('fill',colors[idx]);
  badge.textContent=['κ(x) computed','Fallback: 100 req/s (10%)','Fallback: 220 req/s (22%)','Safe default emitted','Fallback: 380 req/s (38%)'][idx];
  readout.textContent=readouts[idx];
  invariant.textContent='Fallback capacity law: λ_fallback = (1 − coverage(τ)) × λ_arrival must fit secondary model/reviewer SLOs.';
  root.dataset.values=JSON.stringify({step:idx,tau:[0.55,0.55,0.72,0.72,0.85][idx],fallback_rps:[100,100,220,220,380][idx]});
 }

 return window.MLSysOriginalPlayer.create({
  root,outer:context.outer,layout:'compact',
  title:"How does raising threshold \u03c4 trade accepted risk against fallback load?",
  intro:"Route requests across Answer, Route, and Abstain while raising \u03c4 from 0.55 to 0.85 to trade selective risk against 380 req/s fallback load.",
  scope:"Illustrative competence-routing model at 1,000 req/s arrival rate (new-explore-batch.calc.py): threshold settings \u03c4 = 0.55 (90% coverage, 0.082 risk, 100 req/s fallback), \u03c4 = 0.72 (78% coverage, 0.041 risk, 220 req/s fallback), and \u03c4 = 0.85 (62% coverage, 0.019 risk, 380 req/s fallback).",
  steps,
  render(i,{stage,details}){if(!canvas.isConnected)stage.append(canvas);if(details&&!readout.isConnected)details.append(readout,invariant);root.dataset.sceneStep=String(i);paint(i,i,1);},
  transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>{root.dataset.animationProgress=String(p);paint(from,to,p);});}
 });
};})();
