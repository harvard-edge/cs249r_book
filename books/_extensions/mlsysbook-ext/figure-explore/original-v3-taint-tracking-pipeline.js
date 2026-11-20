/* Original-artwork Explore mechanism. Shared player owns the pausable clock. */
(()=>{'use strict';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v3-taint-tracking-pipeline"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),NS='http://www.w3.org/2000/svg';
 const vb=svg.viewBox.baseVal,W=vb.width,H=vb.height;
 if(!W||!H)throw Error('Original artwork viewBox required');
 root.classList.add("original-v3-taint-tracking-pipeline");
 svg.setAttribute('role','img');svg.setAttribute('aria-label',"How do integrity union and confidentiality intersection block exfiltration and fake passes?");
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
 const steps=[{"label": "Tag the 2 planner inputs", "description": "On the upper left, untrusted README carries integrity ι = {readme_7} and open confidentiality κ = every recipient, while Private fixture carries ι = {fixture} and restricted κ = {local test runner}."}, {"label": "Join labels at planner invocation", "description": "Because the blue Planner invocation reads both inputs to form Derived query y, provenance takes the union ι(y) = {readme_7, fixture} while allowed recipients shrink to the intersection κ(y) = {local test runner}."}, {"label": "Block exfiltration at the Release check", "description": "Derived query y enters the Reference monitor's Release check for package_search(y): host-resolved recipient r = search endpoint is not in κ(y) = {local test runner}, so dispatch is rejected before network egress."}, {"label": "Extract an untrusted test claim", "description": "Along the bottom path, Reader extraction x parses claimed_tests_passed: true out of the README, preserving provenance ι(x) = {readme_7} as document content rather than runner output."}, {"label": "Refuse fake pass at the Evidence check", "description": "Recording a test pass requires Inf(o) = {protected runner}; because ι(x) = {readme_7} is not a subset of {protected runner}, the Evidence check refuses to record a pass."}];

 const boxes=[[20,30,170,76,'#DC2626'],[20,124,170,76,'#16A34A'],[215,80,100,70,'#0284C7'],[345,30,200,170,'#7C3AED'],[580,52,190,164,'#DC2626'],[20,268,525,70,'#DC2626'],[580,228,190,150,'#DC2626']];
 const hi=boxes.map(([x,y,w,h,c])=>add('rect',{x:x-2.5,y:y-2.5,width:w+5,height:h+5,rx:5,fill:c,'fill-opacity':'.10',stroke:c,'stroke-width':2.2,visibility:'hidden'}));
 const tokA=add('circle',{r:6,fill:'#0284C7',stroke:'#fff','stroke-width':1.8,'data-motion-object':'true'});
 const tokB=add('circle',{r:6,fill:'#16A34A',stroke:'#fff','stroke-width':1.8,'data-motion-object':'true',visibility:'hidden'});
 const stamp=add('text',{x:282,y:376,'text-anchor':'middle','font-size':11,'font-weight':'700',fill:'#7C3AED',stroke:'#fff','stroke-width':3.5,'paint-order':'stroke'},'');
 const active=[[0,1],[2,3],[3,4],[5],[5,6]];
 const readouts=[
  'Inputs labeled: README (ι={readme_7}, κ=all) + Private fixture (ι={fixture}, κ={local test runner})',
  'Derived query y: ι(y) = ι₁ ∪ ι₂ = {readme_7, fixture} · κ(y) = κ₁ ∩ κ₂ = {local test runner}',
  'Release check on package_search(y): search endpoint ∉ {local test runner} → REJECTED before dispatch',
  'Reader extraction x: claimed_tests_passed: true retains untrusted provenance ι(x) = {readme_7}',
  'Evidence check on test pass: {readme_7} ⊈ Inf(o)={protected runner} → NOT recorded as a pass'
 ];
 function paint(from,to,p){
  const idx=p<1?from:to,moving=from!==to&&p<1;
  hi.forEach((el,k)=>el.setAttribute('visibility',active[idx].includes(k)?'visible':'hidden'));
  tokA.setAttribute('visibility',moving?'visible':'hidden');
  tokB.setAttribute('visibility','hidden');
  if(from===0&&to===1&&moving){
   tokB.setAttribute('visibility','visible');
   const p1=routePt([[105,68],[265,115],[445,115]],p),p2=routePt([[105,162],[265,115],[445,158]],p);
   tokA.setAttribute('cx',p1[0]);tokA.setAttribute('cy',p1[1]);tokA.setAttribute('fill','#DC2626');
   tokB.setAttribute('cx',p2[0]);tokB.setAttribute('cy',p2[1]);
  }else if(idx===1){
   tokA.setAttribute('cx',445);tokA.setAttribute('cy',130);tokA.setAttribute('fill','#7C3AED');
  }else if(idx===2||to===2){
   const [x,y]=routePt([[445,130],[675,130],[675,190]],from===1&&to===2?p:1);
   tokA.setAttribute('cx',x);tokA.setAttribute('cy',y);tokA.setAttribute('fill','#DC2626');
  }else if(idx===3||to===3){
   const [x,y]=routePt([[105,303],[357,303]],from===2&&to===3?p:1);
   tokA.setAttribute('cx',x);tokA.setAttribute('cy',y);tokA.setAttribute('fill','#D97706');
  }else{
   const [x,y]=routePt([[357,303],[675,303],[675,350]],from===3&&to===4?p:1);
   tokA.setAttribute('cx',x);tokA.setAttribute('cy',y);tokA.setAttribute('fill','#DC2626');
  }
  stamp.setAttribute('fill',(idx===2||idx===4)?'#DC2626':'#7C3AED');
  stamp.textContent=['2 labeled sources tagged','Label join: ι union, κ intersection','Egress blocked: r ∉ κ(y)','Untrusted claim isolated as data','Integrity blocked: ι(x) ⊈ Inf(o)'][idx];
  readout.textContent=readouts[idx];
  invariant.textContent='Dual lattice law: confidentiality intersects (κ = ∩ κᵢ) to block leaks; integrity unions (ι = ∪ ιᵢ) to block untrusted authority.';
  root.dataset.values=JSON.stringify({step:idx,release_allowed:false,evidence_accepted:false});
 }

 return window.MLSysOriginalPlayer.create({
  root,outer:context.outer,layout:'compact',
  title:"How do integrity union and confidentiality intersection block exfiltration and fake passes?",
  intro:"Propagate integrity and confidentiality labels across 2 inputs to see why the reference monitor blocks both private-fixture exfiltration and unverified test claims.",
  scope:"Dual-label information-flow control on the repair task: untrusted README (\u03b9 = {readme_7}, \u03ba = every recipient), secret Private fixture (\u03b9 = {fixture}, \u03ba = {local test runner}), derived planner query y, and reader-extracted test assertion x.",
  steps,
  render(i,{stage,details}){if(!canvas.isConnected)stage.append(canvas);if(details&&!readout.isConnected)details.append(readout,invariant);root.dataset.sceneStep=String(i);paint(i,i,1);},
  transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>{root.dataset.animationProgress=String(p);paint(from,to,p);});}
 });
};})();
