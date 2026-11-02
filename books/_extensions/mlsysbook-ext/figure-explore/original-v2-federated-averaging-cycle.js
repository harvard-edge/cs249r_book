(() => {
'use strict';
const NS='http://www.w3.org/2000/svg';
const el=(tag,a={},text)=>{const n=document.createElementNS(NS,tag);Object.entries(a).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;return n;};
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v2-federated-averaging-cycle"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified source artwork and shared player required');
 if(context.verifiedArtwork.getAttribute('viewBox')!=="0 0 920 450")throw Error('Source geometry changed; review required');
 root.classList.add('v2c-lesson');
 const art=el('svg',{viewBox:"0 0 735 435",role:'img','aria-label':"Whose model enters this FedAvg round?"});
 const base=el('g'),moving=el('g',{'aria-hidden':'true'});art.append(base,moving);
 const text=(x,y,t,a={})=>el('text',{x,y,'font-size':14,fill:'#14294e',...a},t);
 const box=(x,y,w,h,label,fill='#edf2fa',stroke='#45699b')=>{const g=el('g');g.append(el('rect',{x,y,width:w,height:h,rx:2,fill,stroke}),text(x+w/2,y+h/2+5,label,{'text-anchor':'middle','font-weight':600}));return g;};
 const line=(pts,color='#8b9bb0',dash='')=>{const g=el('g'),a=pts.at(-2),b=pts.at(-1),theta=Math.atan2(b[1]-a[1],b[0]-a[0]),tip=[b,[b[0]-8*Math.cos(theta)+4*Math.sin(theta),b[1]-8*Math.sin(theta)-4*Math.cos(theta)],[b[0]-8*Math.cos(theta)-4*Math.sin(theta),b[1]-8*Math.sin(theta)+4*Math.cos(theta)]];g.append(el('polyline',{points:pts.map(p=>p.join(',')).join(' '),fill:'none',stroke:color,'stroke-width':2,'stroke-dasharray':dash}),el('polygon',{points:tip.map(p=>p.join(',')).join(' '),fill:color}));return g;};
 const token=(x,y,label,color='#edf2fa',attrs={})=>{const g=el('g',{'data-object':label,...attrs}),w=Math.max(28,label.length*7+10);g.append(el('rect',{x:x-w/2,y:y-12,width:w,height:24,rx:2,fill:color,stroke:'#45699b'}),text(x,y+5,label,{'text-anchor':'middle','font-size':12,'font-weight':700}));return g;};
 const status=document.createElement('p');status.className='v2c-status';status.setAttribute('aria-live','polite');
 const note=document.createElement('p');note.className='v2c-reading';note.textContent="The coordinator accepts completed model uploads, never raw client data. The toy scalar average illustrates sample weighting; round-selection and deadline policy are declared for this example.";
 note.textContent+=' On narrow screens, the view follows the active transfer; pause to swipe across the full diagram.';
 let mounted=false,stageNode=null,stageResize=null,cameraTarget=0;
 function camera(x){cameraTarget=x;if(!stageNode)return;const scale=art.getBoundingClientRect().width/art.viewBox.baseVal.width;stageNode.scrollLeft=Math.max(0,x*scale-stageNode.clientWidth/2);}
 const ends=[440,495,420,535,690];const counts=[1,2,1,2,3],models=[1,3,5,7,9],accepted=[0,1,2,3];const n=accepted.reduce((s,k)=>s+counts[k],0),sum=accepted.reduce((s,k)=>s+counts[k]*models[k],0),mean=sum/n;
 const controller=window.MLSysOriginalPlayer.create({root,outer:context.outer,title:"Whose model enters this FedAvg round?",scope:"The original coordinator and five-client round. Declared toy scalar models \u03b8k = 1, 3, 5, 7, 9 and example counts nk = 1, 2, 1, 2, 3 illustrate the weighting rule; these are not measurements.",steps:[{"label": "Select and broadcast", "description": "The coordinator selects all five illustrated clients and broadcasts the same initial model \u03b8 = 0. Private data stays on each client."}, {"label": "Local work", "description": "Clients train on their own data. Four clients finish within this illustrated round; client 5 is still running. Local completion times are schematic."}, {"label": "Timely uploads", "description": "Clients 1\u20134 upload their completed models before the deadline. The accepted models and their example counts are held by the coordinator."}, {"label": "Close the round", "description": "The deadline passes. Client 5 is excluded from this round, so its three examples do not enter the aggregation denominator."}, {"label": "Example-weighted aggregate", "description": "The coordinator multiplies each accepted model by its accepted example count, sums these contributions, and divides by the accepted example total. The result becomes the next global model."}].map((step,i)=>({label:step.label,get description(){return Number(root.dataset.modelState)===i?step.description:"On arrival: "+step.description;}})),
 render(index,{stage,details}){
  if(!mounted){stageNode=stage;stage.classList.add('v2c-stage');stage.tabIndex=0;stage.setAttribute('aria-label','Mechanism diagram; swipe horizontally on a narrow screen or use Expand');stage.append(art);details.append(status,note);mounted=true;if(typeof ResizeObserver==='function'){stageResize=new ResizeObserver(()=>camera(cameraTarget));stageResize.observe(stage);}}
  base.replaceChildren();moving.replaceChildren();root.dataset.modelState=String(index);
  root.dataset.accepted=index>=2?'1,2,3,4':'';root.dataset.denominator=String(index>=3?n:0);root.dataset.weightedSum=String(index===4?sum:0);root.dataset.aggregate=index===4?String(mean):'';
base.append(box(index===4?200:185,20,index===4?300:520,index===4?84:68,'','#f4efff','#8557c7'),text(index===4?215:200,43,'Coordinator',{'font-weight':700,fill:'#6d28d9','data-fed-header':'title'}),text(index===4?215:200,68,index===4?`Weighted sum = ${accepted.map(k=>`${counts[k]}×${models[k]}`).join(' + ')} = ${sum}`:index>=2?'Accepted: '+accepted.map(k=>`C${k+1}: θ=${models[k]}, n=${counts[k]}`).join(' · '):'Global θ = 0 → selected cohort',{'font-size':13,'data-fed-header':'formula'}));
if(index===4)base.append(text(215,87,`Next θ = ${sum} / ${n} ≈ ${mean.toFixed(2)}`,{'font-size':13,'data-fed-header':'result'}));
for(let k=0;k<5;k++){const y=126+k*54;const rejected=k===4&&index>=3;base.append(text(15,y+5,`Client ${k+1}`,{'font-weight':700}),box(95,y-16,65,33,`D${k+1}: ${counts[k]}`,'#dcfce7','#16a34a'),box(200,y-17,ends[k]-200,34,'','#edf2fa'),text(210,y+5,index===0?'θ = 0 received':k===4&&index<3?'local training…':`local θ${k+1} = ${models[k]}`,{'font-size':13}),line([[160,y],[200,y]],'#94a3b8'));
if(index>=2&&k<4)base.append(token(ends[k]+20,y,`θ${k+1}`,'#fff3cd',{'data-accepted-client':k+1}),line([[ends[k]+40,y],[580,y],[580,88]],'#d97706'));
if(rejected)base.append(el('rect',{x:600,y:y-17,width:ends[k]-600,height:34,fill:'#fee2e2',stroke:'#ef4444','stroke-dasharray':'3 2','data-late-region':'true'}),text(610,y+5,'excluded',{fill:'#b91c1c','font-weight':700,'data-excluded-client':'5'}));}
base.append(line([[600,102],[600,395]],'#ef4444','5 4'),text(600,416,'round deadline',{'text-anchor':'middle',fill:'#b91c1c'}));
status.textContent=index===4?`Accepted examples: ${counts.slice(0,4).join(' + ')} = ${n}. Weighted model sum: ${sum}. Next θ (rounded): ${mean.toFixed(2)}. Client 5 contributes neither model nor examples.`:index>=3?`Deadline closed: clients 1–4 accepted; client 5 excluded. Denominator: ${n} accepted examples.`:index===2?'Coordinator holds four timely uploads; client 5 has not completed.':'Private data D₁–D₅ remains on the clients; only model copies cross the network.';
 camera(index===0?410:index===1?320:index===4?350:550);
 },
 transition({from,to,onFrame}){
  if(to!==from+1)return;
  const motions=to===1?counts.map((_,k)=>({label:k===4?'local':`θ${k+1}`,path:[[220,126+k*54],[Math.min(ends[k]-20,560),126+k*54]]})):to===2?accepted.map(k=>({label:`θ${k+1}`,path:[[ends[k],126+k*54],[ends[k]+20,126+k*54],[580,126+k*54],[580,75]]})):to===3?[{label:'local',color:'#fee2e2',path:[[560,342],[660,342]]}]:to===4?accepted.map(k=>({label:`${counts[k]}×${models[k]}`,path:[[ends[k]+20,126+k*54],[580,126+k*54],[580,75],[450,75]]})):[];
  status.textContent="Transition in progress; destination state is pending. "+status.textContent;
  if(motions.length===16)base.querySelectorAll("[data-live-token]").forEach(n=>n.setAttribute("visibility","hidden"));
  const packets=motions.map(m=>{const g=token(0,0,m.label,m.color||'#fff3cd',{'data-motion-object':'true'});moving.append(g);const lengths=m.path.slice(1).map((p,i)=>Math.hypot(p[0]-m.path[i][0],p[1]-m.path[i][1]));return {...m,g,lengths,total:lengths.reduce((a,b)=>a+b,0)};});
  onFrame(p=>{root.dataset.animationProgress=String(p);packets.forEach(m=>{let d=p*m.total,i=0;while(i<m.lengths.length-1&&d>m.lengths[i])d-=m.lengths[i++];const a=m.path[i],b=m.path[i+1],f=m.lengths[i]?d/m.lengths[i]:1;const px=a[0]+(b[0]-a[0])*f;m.g.setAttribute('transform',`translate(${px},${a[1]+(b[1]-a[1])*f})`);if(m===packets[0])camera(px);});});
 }});
 const destroy=controller.destroy;controller.destroy=()=>{stageResize?.disconnect();destroy();};
 return controller;
};
})();
