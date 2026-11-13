/* Original-artwork Explore mechanism. Shared player owns the pausable clock. */
(()=>{'use strict';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v4-16-release-record"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),NS='http://www.w3.org/2000/svg';
 const vb=svg.viewBox.baseVal,W=vb.width,H=vb.height;
 if(!W||!H)throw Error('Original artwork viewBox required');
 root.classList.add("original-v4-16-release-record");
 svg.setAttribute('role','img');svg.setAttribute('aria-label',"When can stale calibration still reach COND instead of REFUSE?");
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
 const steps=[{"label": "Admit full envelope as UNCOND", "description": "Among the 5 release-gate paths, a valid UNCOND case whose key chain, binary images, complete fault records, and current physical calibrations all pass issues an UNCOND full-envelope permit."}, {"label": "Admit signed restricted envelope as COND", "description": "When all cryptographic and physical standing checks are valid and current for a restricted release case, the gate issues COND for that signed restricted envelope."}, {"label": "Refuse authentication or fault-log mismatch", "description": "Any expired signature, modified binary image, or incomplete fault-record log fails at Authenticate and drops directly to REFUSE (permit off, assessed stop or hold)."}, {"label": "Refuse uncovered stale calibration", "description": "If Authenticate passes but a sensor calibration is stale at Check physical standing—and no separately signed COND case covers that sensor—the gate takes the 'no' branch into REFUSE."}, {"label": "Admit stale calibration only via covering COND", "description": "Only when a separately signed COND case explicitly locks out dependent motions or derates speed for that named stale sensor does the 'yes' branch issue COND—and every action is still checked by the 1 kHz permission path."}];

 const rects=[[20,20,190,58,'#5B4B8A'],[245,20,190,58,'#1A4D3E'],[470,20,190,58,'#007A87'],[290,120,145,50,'#B87333'],[330,205,130,54,'#1A4D3E'],[490,205,130,54,'#1A4D3E'],[20,290,640,46,'#A51C30']];
 const hi=rects.map(([x,y,w,h,c])=>add('rect',{x:x-2.5,y:y-2.5,width:w+5,height:h+5,rx:4,fill:c,'fill-opacity':'.12',stroke:c,'stroke-width':2.2,visibility:'hidden'}));
 const tok=add('circle',{r:6.5,fill:'#1A4D3E',stroke:'#fff','stroke-width':2,'data-motion-object':'true'});
 const verdict=add('text',{x:145,y:242,'text-anchor':'middle','font-size':11,'font-weight':'700',fill:'#1A4D3E',stroke:'#fff','stroke-width':3.5,'paint-order':'stroke'},'');
 const active=[[0,1,2,5],[0,1,2,4],[0,1,6],[0,1,2,3,6],[0,1,2,3,4]];
 const routes=[
  [[115,49],[340,49],[565,49],[555,232]],
  [[115,49],[340,49],[565,49],[395,232]],
  [[115,49],[340,49],[240,313]],
  [[115,49],[340,49],[565,49],[362,145],[300,313]],
  [[115,49],[340,49],[565,49],[362,145],[395,232]]
 ];
 const cols=['#1A4D3E','#007A87','#A51C30','#A51C30','#B87333'];
 const readouts=[
  'Path 1 / 5 (UNCOND): Auth PASS + standing current & valid → UNCOND full-envelope permit issued',
  'Path 2 / 5 (Signed COND): Auth PASS + standing current & valid → COND restricted-envelope permit issued',
  'Path 3 / 5 (Auth failure): Key/image/fault-log mismatch or expired case → REFUSE · permit off',
  'Path 4 / 5 (Uncovered stale cal): Calibration stale + Separately signed COND = NO → REFUSE · permit off',
  'Path 5 / 5 (Covered stale cal): Calibration stale + Separately signed COND = YES → COND restricted permit'
 ];
 function paint(from,to,p){
  const idx=p<1?from:to,moving=from!==to&&p<1;
  hi.forEach((el,k)=>el.setAttribute('visibility',active[idx].includes(k)?'visible':'hidden'));
  const r=from!==to?routes[to]:routes[idx],q=from!==to?p:1,[x,y]=routePt(r,q);
  tok.setAttribute('visibility',moving?'visible':'hidden');
  tok.setAttribute('cx',x);tok.setAttribute('cy',y);tok.setAttribute('fill',cols[idx]);
  verdict.setAttribute('fill',cols[idx]);
  verdict.textContent=['Result: UNCOND','Result: COND','Result: REFUSE','Result: REFUSE','Result: COND (derated)'][idx];
  readout.textContent=readouts[idx];
  invariant.textContent='Fail-closed release rule: every unauthenticated, unknown, or uncovered stale condition ends in REFUSE; runtime permission checks remain active.';
  root.dataset.values=JSON.stringify({step:idx,gate_result:['UNCOND','COND','REFUSE','REFUSE','COND'][idx]});
 }

 return window.MLSysOriginalPlayer.create({
  root,outer:context.outer,layout:'compact',
  title:"When can stale calibration still reach COND instead of REFUSE?",
  intro:"Evaluate all 5 release-gate paths across cryptographic authentication and physical standing checks to see when a case earns UNCOND, COND, or REFUSE.",
  scope:"Fail-closed startup and operating-permit release gate (Table 16.1): Signed release case, key/image/fault-record authentication, physical standing check (calibration, brakes, envelope, age), and separate restricted COND coverage.",
  steps,
  render(i,{stage,details}){if(!canvas.isConnected)stage.append(canvas);if(details&&!readout.isConnected)details.append(readout,invariant);root.dataset.sceneStep=String(i);paint(i,i,1);},
  transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>{root.dataset.animationProgress=String(p);paint(from,to,p);});}
 });
};})();
