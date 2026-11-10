/* Original figure reviewed; redesigned causal geometry. Shared player owns time. */
(()=>{'use strict';window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v3-vol3-asynchronous-freshness-pipeline"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and player required');
 const NS='http://www.w3.org/2000/svg', make=(t,a={},v)=>{const n=document.createElementNS(NS,t);for(const[k,x]of Object.entries(a))n.setAttribute(k,x);if(v!==undefined)n.textContent=v;return n};
 root.classList.add('v3b-mechanism');const canvas=document.createElement('div');canvas.className='v3b-canvas';canvas.tabIndex=0;canvas.setAttribute('aria-label','Mechanism diagram; scroll horizontally to inspect all lanes, or use Expand.');
 let svg;
 const cue=document.createElement('p');cue.className='v3b-cue';cue.textContent='Schematic state transitions; playback speed is not measured time. On a narrow screen, swipe the diagram or use Expand.';
 const readout=document.createElement('p');readout.className='v3b-readout';
 const LANES=[["Rollout slots", 25], ["Verification", 100], ["Group buffer", 175], ["Admission", 270]];
 const palette={compute:['#e0f2fe','#0284c7'],memory:['#dcfce7','#16a34a'],system:['#f3e8ff','#7c3aed'],tool:['#fef3c7','#d97706'],fault:['#fee2e2','#dc2626'],neutral:['#f8fafc','#94a3b8']};
 const box=(id,x,y,w,label,kind='neutral',h=38)=>({id,x,y,w,h,label,kind});
 function model(i){const boxes=[];if(i<1)boxes.push(box('A',145,25,120,'A @ v0','compute'));else boxes.push(box('A',i===1?145:i<4?145:145,i===1?100:i<4?175:270,150,i>=4?'A behavior v0|check v1 · lag '+(1-0):'A @ v0 · '+(i>=2?'pass':'pending'),'memory'));if(i<3)boxes.push(box('B',325,25,120,'B @ v1','compute'));else boxes.push(box('B',325,i<4?175:270,150,i>=4?'B behavior v1|check v3 · lag '+(3-1):'B @ v1 · fail','memory'));if(i<5)boxes.push(box('C',505,25,120,'C @ v0','compute'));else boxes.push(box('C',505,i===5?100:270,180,i===5?'C @ v0 · verify':'C behavior v0|v4 lag '+(4-0)+' > 2 · exclude',i===5?'tool':'fault'));const results=[
'Active slots: 3 (A @ v0, B @ v1, C @ v0) | Group buffer: 0 records | Synchronous Panel (a) reference wastes 65% of slot time waiting for the 16.8 s straggler.',
'Active slots: 2 (B @ v1, C @ v0); A released slot early | Verification: A @ v0 running | Group buffer: 0 records.',
'Active slots: 2 (B @ v1, C @ v0) | Verification: A resolved (pass) | Group buffer: 1 record (A @ v0 awaiting admission check).',
'Active slots: 1 (C @ v0); B released slot | Verification: B resolved (fail reward is valid training signal) | Group buffer: 2 records (A @ v0, B @ v1).',
'Admission lag check (max lag = 2): A checked at v1 has lag 1 - 0 = 1 <= 2 (admit); B checked at v3 has lag 3 - 1 = 2 <= 2 (admit) | C still running.',
'Active slots: 0 (C released slot at trainer v4) | Verification: C @ v0 running | Admitted records: A (lag 1), B (lag 2).',
'Admission lag check at v4: C sampled at v0 has lag 4 - 0 = 4 > 2 (excluded, reason and cost recorded) | Passing per-record lag checks for A and B does not by itself close the planned cohort.'
];return{boxes,notes:[[145,235,'Bounded compact records; release of a slot does not admit a group.']],result:results[i],invariant:'lag = trainer version − behavior version'};}
 const steps=[
  ["Pin behavior checkpoints", "While Panel (a) shows why synchronous barriers waste 65% of slot capacity across 0.2 s, 2.1 s, 4.2 s, and 16.8 s rollouts, Panel (b) starts 3 asynchronous rollouts pinned to their generating policy versions: A at v0, B at v1, and C at v0."],
  ["A completes and releases slot", "When rollout A finishes generation, it immediately releases its inference and sandbox slot for new work instead of waiting at a barrier, while its compact record (stamped with behavior version v0) enters verification in Panel (c)."],
  ["Verify A and buffer", "Rollout A completes verification with a pass verdict and enters a slot in the bounded group buffer (Panel c), holding its v0 behavior tag until the admission gate evaluates version lag."],
  ["B completes and verifies", "Rollout B (sampled at v1) finishes generation, frees its rollout slot, and completes verification with a resolved fail verdict (which supplies a valid negative reward signal, unlike an unresolved infrastructure timeout) while C continues running."],
  ["Check A and B at their completion versions", "At the admission diamond (delta_ver <= delta_ver_max = 2), rollout A checked at trainer version v1 has lag 1 - 0 = 1 <= 2 (admit), and rollout B checked at v3 has lag 3 - 1 = 2 <= 2 (admit). These per-record checks do not yet close the group."],
  ["C completes at trainer v4", "Long-running rollout C finally finishes after the trainer has advanced to version v4 (ver + 4 in Panel b), releasing its slot and entering verification still carrying its original generating version v0."],
  ["Reject stale C", "When C reaches the admission diamond at trainer version v4, its policy staleness is delta_ver = 4 - 0 = 4, which exceeds the maximum lag bound of 2 updates. The gate routes C to excluded (recording the staleness reason and compute cost), leaving cohort closure policy to decide whether the group updates without C."]
 ].map(([label,description])=>({label,description}));
 let currentNodes=new Map(), currentScene=null;
 const template=context.verifiedArtwork.cloneNode(true);let overlay;
 function rect(x,y){const n=[...svg.querySelectorAll('rect')].find(n=>Number(n.getAttribute('x'))===x&&Number(n.getAttribute('y'))===y);if(!n)throw Error('Unreviewed original rectangle '+x+','+y);return n;}
 function text(value,y,optional=false){const n=[...svg.querySelectorAll('text')].find(n=>n.textContent===value&&(y===undefined||Number(n.getAttribute('y'))===y));if(!n&&!optional)throw Error('Unreviewed original label '+value);return n;}
 const chip=(id,cx,cy,label,kind='neutral',w=28,h=20)=>({id,x:cx-w/2,y:cy-h/2,w,h,label,kind});
 const bar=(id,x,y,w,h,kind)=>({id,x,y,w,h,label:'',kind,bar:true});
 function scene(i){
 // Panel (a) remains the original four-member synchronous reference.
 // Panels (b)/(c) retain original behavior versions, lag decisions, and routes.
 for(const[y,start]of [[284,1],[316,3],[348,5]])rect(y===316?191.5:101.3,y).setAttribute('fill-opacity',i>=start?.16:0);
 for(const[y,start]of [[297,4],[329,4],[361,6]]){const n=[...svg.querySelectorAll('text')].find(n=>Number(n.getAttribute('x'))===528&&Number(n.getAttribute('y'))===y);n.setAttribute('opacity',0);}
 for(const r of [...svg.querySelectorAll('rect')].filter(n=>Number(n.getAttribute('y'))===481))r.setAttribute('fill-opacity',0);
 const items=[];
 if(i>=2&&i<4)items.push(chip('bufferA',303,486,'A','system',10,10));
 if(i>=3&&i<4)items.push(chip('bufferB',317,486,'B','system',10,10));
 if(i>=1)items.push(chip('A',i===1?209:i<4?310:548,i===1?513:i<4?513:297,i>=4?'A · v0 · lag 1':'A · v0','memory',i>=4?88:54,14));
 if(i>=3)items.push(chip('B',i<4?366:548,i<4?513:329,i>=4?'B · v1 · lag 2':'B · v1','memory',i>=4?88:54,14));
 if(i>=5)items.push(chip('C',i===5?209:462,i===5?513:568,i===5?'C · v0':'C · v0 · lag 4 ×',i===5?'tool':'fault',i===5?62:154,16));
 return{items,focus:i>=5?'C':i>=3?'B':i>=1?'A':null,cue:'Panel (a) is the synchronous reference. Live record identities use the original (b)/(c) paths; passing a lag check does not admit a complete group.'};
}
 function route(from,to,id){return null;}
 function draw(item){const [fill,stroke]=palette[item.kind];const g=make('g',{'data-object':item.id,transform:`translate(${item.x} ${item.y})`});g.append(make('rect',{width:item.w,height:item.h,rx:item.bar?0:2,fill,'fill-opacity':item.bar?.36:1,stroke:item.bar?'none':stroke,'stroke-width':1}));if(item.label)g.append(make('text',{x:item.w/2,y:item.h/2+4,'text-anchor':'middle',class:'v3b-label',style:'font-size:'+Math.min(11,item.h-2)+'px'},item.label));overlay.append(g);currentNodes.set(item.id,g);return g;}
 function follow(item){if(!item)return;const vb=svg.viewBox.baseVal,scale=svg.getBoundingClientRect().width/vb.width;canvas.scrollLeft=Math.max(0,(item.x+item.w/2)*scale-canvas.clientWidth/2);if(canvas.scrollHeight>canvas.clientHeight)canvas.scrollTop=Math.max(0,(item.y+item.h/2-vb.y)*scale-canvas.clientHeight/2);}
 function paint(i){svg=template.cloneNode(true);svg.classList.add('v3b-original-art');svg.setAttribute('role','img');svg.setAttribute('aria-label',"Which record is still fresh at admission?"+' — original artwork with live identity/state overlays');canvas.replaceChildren(svg);overlay=make('g',{'data-live-overlay':'true'});svg.append(overlay);currentNodes=new Map();const m=model(i),s=scene(i);s.items.forEach(draw);currentScene=s;readout.textContent=m.result;root.dataset.mechanismState=String(i);root.dataset.objects=JSON.stringify(s.items.map(b=>b.id));root.dataset.facts=JSON.stringify(m);root.dataset.invariant=m.invariant||'';root.dataset.originalViewBox=template.getAttribute('viewBox');root.dataset.originalChildCount=String(template.children.length);const focus=s.items.find(x=>x.id===s.focus)||s.items[0];cue.textContent=(s.cue||'')+' Live committed state: '+steps[i].label+'. Swipe the original diagram or use Expand.';if(focus)follow(focus);else{canvas.scrollLeft=0;canvas.scrollTop=0;}return s;}
 function render(i,{stage,details}){if(!canvas.isConnected){stage.append(cue,canvas);details.append(readout);}paint(i);if(!details.querySelector('.v3-original-guard')){const p=document.createElement('p');p.className='v3-original-guard';p.textContent="Toy version differences are computed from declared integers. Verification can pass, fail, or be unavailable; a resolved failed reward is not infrastructure failure. Per-record freshness cannot replace planned-cohort closure. Update-count lag does not bound parameter-space drift.";details.append(p);}}
 function transition({from,to,onFrame}){if(to!==from+1)return;const a=currentScene,old=new Map(a.items.map(x=>[x.id,x]));
 // Build the destination on a detached verified clone to obtain its live geometry,
 // then restore source DOM and committed facts while the shared clock moves objects.
 const sourceSvg=svg,sourceOverlay=overlay,sourceNodes=currentNodes,sourceCue=cue.textContent;const destinationSvg=template.cloneNode(true);svg=destinationSvg;overlay=make('g');svg.append(overlay);const b=scene(to);svg=sourceSvg;overlay=sourceOverlay;currentNodes=sourceNodes;
 const movements=[];for(const target of b.items){const source=old.get(target.id);let n=currentNodes.get(target.id);if(!n){n=draw({...target,label:target.label});n.setAttribute('opacity',0);}movements.push({n,source,target,route:route(from,to,target.id)});n.dataset.motionObject='true';}
 const active=b.items.find(t=>{const s=old.get(t.id);return s&&(s.x!==t.x||s.y!==t.y||s.w!==t.w);})||b.items.find(t=>!old.has(t.id))||b.items.find(t=>t.id===b.focus);
 cue.textContent='Pending action: '+steps[to].label+'. '+(b.cue||'');
 onFrame(p=>{root.dataset.animationProgress=String(p);for(const{n,source,target,route}of movements){if(source){let x=source.x+(target.x-source.x)*p,y=source.y+(target.y-source.y)*p;if(route){const q=p*(route.length-1),k=Math.min(route.length-2,Math.floor(q)),f=q-k;x=route[k][0]+(route[k+1][0]-route[k][0])*f-target.w/2;y=route[k][1]+(route[k+1][1]-route[k][1])*f-target.h/2;}n.setAttribute('transform',`translate(${x} ${y})`);n.querySelector('rect').setAttribute('width',source.w+(target.w-source.w)*p);}else n.setAttribute('opacity',String(p));}if(active){const track=movements.find(m=>m.target.id===active.id),matrix=track?.n.transform.baseVal.consolidate()?.matrix,r=track?.n.querySelector('rect');if(matrix&&r)follow({x:matrix.e,y:matrix.f,w:Number(r.getAttribute('width')),h:Number(r.getAttribute('height'))});}readout.textContent=model(from).result+' Destination state remains pending until the movement completes.';});}
 return window.MLSysOriginalPlayer.create({root,outer:context.outer,title:"Which record is still fresh at admission?",intro:"Trace 3 asynchronous rollouts against a maximum policy lag of 2 trainer updates to see which completed records enter the group buffer and which stale rollout is excluded.",layout:"compact",scope:"Three-panel RLVR freshness pipeline: Panel (a) shows a synchronous 4-rollout group (0.2 s, 2.1 s, 4.2 s, and 16.8 s) where waiting for the 16.8 s straggler leaves 65% of reserved slot time idle. Panels (b) and (c) trace 3 asynchronous rollouts (A and C sampled at behavior version v0, B at v1) checked at trainer versions v1, v3, and v4 against a declared maximum policy lag of 2 updates.",steps,render,transition});};})();
