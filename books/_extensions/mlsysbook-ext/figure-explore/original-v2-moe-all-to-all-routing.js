(() => {
'use strict';
const NS='http://www.w3.org/2000/svg';
const el=(tag,a={},text)=>{const n=document.createElementNS(NS,tag);Object.entries(a).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;return n;};
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v2-moe-all-to-all-routing"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified source artwork and shared player required');
 if(context.verifiedArtwork.getAttribute('viewBox')!=="0 0 760 438")throw Error('Source geometry changed; review required');
 root.classList.add('v2c-lesson');
 const art=el('svg',{viewBox:"0 0 760 425",role:'img','aria-label':"How does each expert result find its original token?"});
 const base=el('g'),moving=el('g',{'aria-hidden':'true'});art.append(base,moving);
 const text=(x,y,t,a={})=>el('text',{x,y,'font-size':14,fill:'#14294e',...a},t);
 const box=(x,y,w,h,label,fill='#edf2fa',stroke='#45699b')=>{const g=el('g');g.append(el('rect',{x,y,width:w,height:h,rx:2,fill,stroke}),text(x+w/2,y+h/2+5,label,{'text-anchor':'middle','font-weight':600}));return g;};
 const line=(pts,color='#8b9bb0',dash='')=>{const g=el('g'),a=pts.at(-2),b=pts.at(-1),theta=Math.atan2(b[1]-a[1],b[0]-a[0]),tip=[b,[b[0]-8*Math.cos(theta)+4*Math.sin(theta),b[1]-8*Math.sin(theta)-4*Math.cos(theta)],[b[0]-8*Math.cos(theta)-4*Math.sin(theta),b[1]-8*Math.sin(theta)+4*Math.cos(theta)]];g.append(el('polyline',{points:pts.map(p=>p.join(',')).join(' '),fill:'none',stroke:color,'stroke-width':2,'stroke-dasharray':dash}),el('polygon',{points:tip.map(p=>p.join(',')).join(' '),fill:color}));return g;};
 const token=(x,y,label,color='#edf2fa',attrs={})=>{const g=el('g',{'data-object':label,...attrs}),w=Math.max(28,label.length*7+10);g.append(el('rect',{x:x-w/2,y:y-12,width:w,height:24,rx:2,fill:color,stroke:'#45699b'}),text(x,y+5,label,{'text-anchor':'middle','font-size':12,'font-weight':700}));return g;};
 const status=document.createElement('p');status.className='v2c-status';status.setAttribute('aria-live','polite');
 const note=document.createElement('p');note.className='v2c-reading';note.textContent="Trace amber token a. The other fifteen tokens move in the same exchange. Geometry is schematic and encodes routing, not measured latency.";
 note.textContent+=' On narrow screens, the view follows the active transfer; pause to swipe across the full diagram.';
 let mounted=false,stageNode=null;
 function camera(x){if(!stageNode)return;const scale=art.getBoundingClientRect().width/art.viewBox.baseVal.width;stageNode.scrollLeft=Math.max(0,x*scale-stageNode.clientWidth/2);}
 const labels='abcdefghijklmnop'.split(''),assignment=[2,0,3,1,1,3,0,2,0,2,1,3,3,1,2,0];const expertRows=Array.from({length:4},(_,r)=>labels.filter((_,i)=>assignment[i]===r));
const origin=i=>[53+(i%4)*33,100+Math.floor(i/4)*78];const expert=i=>[328+expertRows[assignment[i]].indexOf(labels[i])*33,100+assignment[i]*78];const returned=i=>[603+(i%4)*33,100+Math.floor(i/4)*78];
 const controller=window.MLSysOriginalPlayer.create({root,outer:context.outer,title:"How does each expert result find its original token?",scope:"The original sixteen tokens a\u2013p, four ranks, and one expert per rank. Top-1 routing; prime marks an expert output, not a numerical value. One forward pass; backward exchanges are outside this sequence.",steps:[{"label": "Tokens at their owners", "description": "Each rank owns four tokens and records their original positions. The gate assigns one expert to each token."}, {"label": "Dispatch to expert ranks", "description": "All sixteen tokens follow the original top-1 assignment. Expert 2 receives a from rank 0; each expert receives one token from every rank."}, {"label": "Expert computation", "description": "Each expert transforms its assigned tokens. a becomes a\u2032 at Expert 2. The origin rank and token position stay attached to the result."}, {"label": "Inverse return", "description": "The second exchange returns each expert output to its original rank and slot. a\u2032 returns to rank 0, slot a; dispatch destinations are not the output owners."}, {"label": "Combine at origin", "description": "Each original rank applies its gate weight to its returned outputs. Top-1 has one expert result per token; no second expert contribution is invented."}].map((step,i)=>({label:step.label,get description(){return Number(root.dataset.modelState)===i?step.description:"On arrival: "+step.description;}})),
 render(index,{stage,details}){
  if(!mounted){stageNode=stage;stage.classList.add('v2c-stage');stage.tabIndex=0;stage.setAttribute('aria-label','Mechanism diagram; swipe horizontally on a narrow screen or use Expand');stage.append(art);details.append(status,note);mounted=true;}
  base.replaceChildren();moving.replaceChildren();root.dataset.modelState=String(index);
  root.dataset.liveCount=String(labels.length);root.dataset.tokenA=index===0?'rank0:a':index<=2?'expert2': 'rank0:a';root.dataset.valueA=index<2?'a':index===2||index===3?'a-prime':'gate2-times-a-prime';
base.append(text(102,25,'Tokens / origin',{'text-anchor':'middle','font-weight':700}),text(377,25,'Expert FFNs',{'text-anchor':'middle','font-weight':700}),text(652,25,'Return / combine',{'text-anchor':'middle','font-weight':700}),text(239,51,'Dispatch',{'text-anchor':'middle',fill:'#b45309'}),text(517,51,'Return',{'text-anchor':'middle',fill:'#b45309'}));
for(let r=0;r<4;r++){for(const [x,label]of [[25,`Rank ${r}`],[300,`Expert ${r}`],[575,`Rank ${r}`]]){base.append(box(x,65+r*78,155,63,'','#f8fafc'),text(x+10,84+r*78,label,{'font-size':12,'font-weight':700}));}}
labels.forEach((label,i)=>{const a=origin(i),b=expert(i),c=returned(i);base.append(line([[180,a[1]],[300,b[1]]],i===0?'#b45309':'#ead8ad'),line([[455,b[1]],[575,c[1]]],i===0?'#b45309':'#ead8ad'));for(const p of [a,b,c])base.append(el('rect',{x:p[0]-14,y:p[1]-11,width:28,height:23,fill:'#fff',stroke:'#cbd5e1'}));let p=index===0?a:index<=2?b:c;const g=token(p[0],p[1],index===4?'y_'+label:label+(index>=2?'′':''),i===0?'#fff3cd':'#edf2fa',{'data-live-token':'true','data-origin':Math.floor(i/4),'data-slot':i%4,'data-expert':assignment[i],'data-location':index===0?'origin':index<=2?'expert':'returned','data-value':index<2?'input':index===4?'weighted':'expert-output'});base.append(g);});
base.append(text(25,405,index===4?'Combine: y_a = gate₂(a) × a′; each token uses its own recorded gate weight.':'Outlined slots preserve the routing map; filled chips are the live token values.'));
status.textContent=`a · ${index===0?'rank 0, slot a → gate selects Expert 2':index===1?'input a at Expert 2':index===2?'a′ = Expert 2(a), carrying origin rank 0 / slot a':index===3?'a′ returned to rank 0 / slot a':'rank 0 combines gate₂(a) × a′'}. All ${labels.length} identities are conserved.`;
 camera(index===0?105:index<=2?380:655);
 },
 transition({from,to,onFrame}){
  if(to!==from+1)return;
  const motions=to===1||to===3?labels.map((label,i)=>({label:label+(to===3?'′':''),path:to===1?[origin(i),expert(i)]:[expert(i),returned(i)]})):to===2?[{label:'a → a′',path:[expert(0),[expert(0)[0],expert(0)[1]-20],expert(0)]}]:to===4?[{label:'gate×a′',path:[returned(0),[returned(0)[0],returned(0)[1]-20],returned(0)]}]:[];
  status.textContent="Transition in progress; destination state is pending. "+status.textContent;
  if(motions.length===16)base.querySelectorAll("[data-live-token]").forEach(n=>n.setAttribute("visibility","hidden"));
  const packets=motions.map(m=>{const g=token(0,0,m.label,m.color||'#fff3cd',{'data-motion-object':'true'});moving.append(g);const lengths=m.path.slice(1).map((p,i)=>Math.hypot(p[0]-m.path[i][0],p[1]-m.path[i][1]));return {...m,g,lengths,total:lengths.reduce((a,b)=>a+b,0)};});
  onFrame(p=>{root.dataset.animationProgress=String(p);packets.forEach(m=>{let d=p*m.total,i=0;while(i<m.lengths.length-1&&d>m.lengths[i])d-=m.lengths[i++];const a=m.path[i],b=m.path[i+1],f=m.lengths[i]?d/m.lengths[i]:1;const px=a[0]+(b[0]-a[0])*f;m.g.setAttribute('transform',`translate(${px},${a[1]+(b[1]-a[1])*f})`);if(m===packets[0])camera(px);});});
 }});
 return controller;
};
})();
