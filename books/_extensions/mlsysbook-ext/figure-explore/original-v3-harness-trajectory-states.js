/* Original-artwork Explore mechanism. Shared player owns the pausable clock. */
(()=>{'use strict';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v3-harness-trajectory-states"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),NS='http://www.w3.org/2000/svg';
 const vb=svg.viewBox.baseVal,W=vb.width,H=vb.height;
 if(!W||!H)throw Error('Original artwork viewBox required');
 root.classList.add("original-v3-harness-trajectory-states");
 svg.setAttribute('role','img');svg.setAttribute('aria-label',"Why must approvals re-check dispatch guards and all exits traverse FINALIZING?");
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
 const steps=[{"label": "Cycle READY → CALLING_MODEL → WAITING_TOOL", "description": "In the ordinary cycle among the 5 live states, READY verifies admission guards before reserving CALLING_MODEL, dispatches an admitted tool call into WAITING_TOOL, and returns to READY once wait policy is met."}, {"label": "Hold reviewed actions in AWAITING_APPROVAL", "description": "When policy requires human review, CALLING_MODEL transitions to dashed AWAITING_APPROVAL; operator approval advances to WAITING_TOOL only if active dispatch guards still pass at release time."}, {"label": "Return denied or stale approvals to READY", "description": "If an approval request is denied, expires, or becomes stale against newer state, AWAITING_APPROVAL emits a bounded refused observation back to READY without dispatching the tool."}, {"label": "Pause only from quiescent READY", "description": "Operator pause enters PAUSED (no new work) cleanly from READY and resumes back to READY, avoiding mid-flight state corruption."}, {"label": "Drain obligations in FINALIZING", "description": "Completion requests—or cancel, kill, budget, or deadline triggers from any of the 5 live states—enter FINALIZING, which immediately stops new admission and settles outstanding reservations."}, {"label": "Settle in 1 of 3 absorbing outcomes", "description": "Only after FINALIZING resolves or transfers remaining duties does the trajectory enter 1 of the 3 double-bordered absorbing states: COMPLETED, FAILED, or CANCELED."}];

 const stBoxes=[[70,180,160,50,'#0f172a'],[330,180,160,50,'#0284C7'],[70,310,160,50,'#D97706'],[330,310,160,50,'#D97706'],[70,60,160,50,'#64748b'],[580,180,160,50,'#7C3AED'],[620,270,166,50,'#16A34A'],[620,330,166,50,'#DC2626'],[620,390,166,50,'#64748b']];
 const hi=stBoxes.map(([x,y,w,h,c])=>add('rect',{x:x-3,y:y-3,width:w+6,height:h+6,rx:5,fill:c,'fill-opacity':'.12',stroke:c,'stroke-width':2.3,visibility:'hidden'}));
 const cursor=add('circle',{r:7,fill:'#7C3AED',stroke:'#fff','stroke-width':2,'data-motion-object':'true'});
 const note=add('text',{x:270,y:462,'text-anchor':'middle','font-size':11,'font-weight':'700',fill:'#7C3AED',stroke:'#fff','stroke-width':3.5,'paint-order':'stroke'},'');
 const active=[[0,1,2],[1,3,2],[3,0],[0,4],[5],[5,6,7,8]];
 const paths=[
  [[150,205],[410,205],[150,335],[150,205]],
  [[410,205],[410,335],[150,335]],
  [[410,335],[225,400],[150,205]],
  [[150,205],[150,85],[150,205]],
  [[410,205],[660,205]],
  [[660,205],[700,295]]
 ];
 const cols=['#0284C7','#D97706','#DC2626','#64748b','#7C3AED','#16A34A'];
 const readouts=[
  'Ordinary turn loop: READY (guards pass) → CALLING_MODEL → WAITING_TOOL → READY',
  'Approval detour: CALLING_MODEL → AWAITING_APPROVAL → approved + fresh guards pass → WAITING_TOOL',
  'Refusal loopback: denied, expired, or stale approval → refused observation → READY (0 tool calls)',
  'Quiescent control hold: READY ↔ PAUSED (no new dispatches admitted while paused)',
  'Mandatory drain: any live exit → FINALIZING · admission stopped while reservations & leases settle',
  'Absorbing terminal states (3): COMPLETED, FAILED, or CANCELED entered only from FINALIZING'
 ];
 function paint(from,to,p){
  const idx=p<1?from:to,moving=from!==to&&p<1;
  hi.forEach((el,k)=>el.setAttribute('visibility',active[idx].includes(k)?'visible':'hidden'));
  const r=from!==to?paths[to]:paths[idx],q=from!==to?p:1,[x,y]=routePt(r,q);
  cursor.setAttribute('visibility',moving?'visible':'hidden');
  cursor.setAttribute('cx',x);cursor.setAttribute('cy',y);cursor.setAttribute('fill',cols[idx]);
  note.setAttribute('fill',cols[idx]);
  note.textContent=['Guarded live cycle (5 live states)','Approval + re-verified guards','Refused/stale → READY observation','PAUSED entered only from READY','FINALIZING drains open duties','3 absorbing terminal outcomes'][idx];
  readout.textContent=readouts[idx];
  invariant.textContent='Harness invariant: approval alone never bypasses dispatch guards, and no trajectory reaches a terminal state without FINALIZING.';
  root.dataset.values=JSON.stringify({step:idx,live_states:5,terminal_states:3});
 }

 return window.MLSysOriginalPlayer.create({
  root,outer:context.outer,layout:'compact',
  title:"Why must approvals re-check dispatch guards and all exits traverse FINALIZING?",
  intro:"Step through the 5 live harness states, approval guard detour, and FINALIZING drain into the 3 absorbing terminal outcomes.",
  scope:"Agent harness trajectory state machine: 5 live states (READY, PAUSED, CALLING_MODEL, WAITING_TOOL, AWAITING_APPROVAL), mandatory drain state FINALIZING, and 3 double-bordered absorbing states (COMPLETED, FAILED, CANCELED).",
  steps,
  render(i,{stage,details}){if(!canvas.isConnected)stage.append(canvas);if(details&&!readout.isConnected)details.append(readout,invariant);root.dataset.sceneStep=String(i);paint(i,i,1);},
  transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>{root.dataset.animationProgress=String(p);paint(from,to,p);});}
 });
};})();
