/* Original-artwork mechanism. Shared player owns the pausable clock. */
(() => {'use strict';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v4-03-chunk-supply-horizon"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),ns='http://www.w3.org/2000/svg';
 const originalView=svg.getAttribute('viewBox');if(originalView!=="0 0 3606 2881")throw Error('Original geometry changed; retain Diagram for review');
 root.classList.add("mlsys-original-v4-03-chunk-supply-horizon");
 const canvas=document.createElement('div');canvas.className='v4b-canvas';canvas.tabIndex=0;canvas.setAttribute('aria-label','Original artwork. Scroll to inspect; Expand opens a larger view.');canvas.append(svg);
 svg.setAttribute('role','img');svg.setAttribute('aria-label',"Why does buffered supply outlast permission?");
 const overlay=document.createElementNS(ns,'g');svg.append(overlay);
 const add=(tag,attrs={},text)=>{const el=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))el.setAttribute(k,String(v));if(text!=null)el.textContent=text;overlay.append(el);return el;};
 const readout=document.createElement('p');readout.className='v4b-readout';readout.setAttribute('aria-live','off');
 const equation=document.createElement('p');equation.className='v4b-equation';
 const note=document.createElement('p');note.className='v4b-note';note.textContent='Scroll the original artwork to inspect labels, or use Expand. Playback duration is illustrative.';
 const width=+originalView.split(' ')[2],height=+originalView.split(' ')[3];
 function follow(x,y){const scale=svg.getBoundingClientRect().width/width;canvas.scrollLeft=Math.max(0,x*scale-canvas.clientWidth/2);canvas.scrollTop=Math.max(0,y*scale-canvas.clientHeight/2);}
 function mount(stage,details){if(!canvas.isConnected)stage.append(canvas,note);if(!readout.isConnected)details.append(readout,equation);}
 function dot(x,y,color='#007A87',r=5){return add('circle',{cx:x,cy:y,r,fill:color,stroke:'#fff','stroke-width':r/3,'data-motion-object':'true'});}
 function route(token,points,p){const lens=points.slice(1).map((b,i)=>Math.hypot(b[0]-points[i][0],b[1]-points[i][1]));let rem=p*lens.reduce((a,b)=>a+b,0),j=0;while(j<lens.length-1&&rem>lens[j]){rem-=lens[j];j++;}const q=lens[j]?rem/lens[j]:1,a=points[j],b=points[j+1];token.setAttribute('cx',a[0]+(b[0]-a[0])*q);token.setAttribute('cy',a[1]+(b[1]-a[1])*q);}
 const D={"period": 50.0, "latency": 40.0, "step": 25.0, "span": 400.0, "replacement": 90.0, "horizon": 219.08902300206645, "slip": 0.5, "initial": 3.0, "tolerance": 15.0, "min_supply": 4};

 const X=t=>616+(3478-616)*t/400;
 const cursor=add('line',{x1:X(0),x2:X(0),y1:850,y2:2410,stroke:'#007A87','stroke-width':9,'stroke-dasharray':'20 14','data-motion-object':'true'});
 const used=add('line',{x1:X(40),x2:X(40),y1:1720,y2:1720,stroke:'#1A4D3E','stroke-width':22,'data-motion-object':'true'});
 const supply=add('line',{x1:X(0),x2:X(0),y1:2080,y2:2080,stroke:'#B87333','stroke-width':17,'data-motion-object':'true'});
 // The source grid is redrawn at the same 25 ms positions to expose ownership.
 // 0..375 are the 16 predicted setpoints; the 400 ms tick is support end.
 add('rect',{x:570,y:2195,width:2950,height:210,fill:'#F4F5F8'});
 const ticks=Array.from({length:16},(_,j)=>({time:j*D.step,mark:add('rect',{x:X(j*D.step)-8,y:2207,width:16,height:190,fill:'#1A4D3E','data-setpoint':j}),slash:add('path',{d:`M ${X(j*D.step)-23} 2245 L ${X(j*D.step)+23} 2355`,stroke:'#A51C30','stroke-width':10,display:'none'})}));
 add('rect',{x:X(D.span)-8,y:2207,width:16,height:190,fill:'#4A5568','data-support-end':'true'});
 const handoff=dot(X(50),2290,'#B87333',27);handoff.style.display='none';
 const times=[0,D.latency,65,D.replacement,200,D.horizon,D.span];
 const steps=[
 ['Capture chunk 0 evidence','The capture clock starts before inference completes. Sixteen capture-relative setpoints span 400 ms; the elapsed prefix cannot be executed retroactively.'],
 ['Make chunk 0 available','Follow inference to its 40 ms endpoint. Only then can chunk 0 enter the direct-replacement stream under the stated assumption of no additional admission delay. Supply alone does not confer permission.'],
 ['Execute within the replacement interval','The active first-chunk segment grows from 40 ms toward 90 ms. Its evidence age is measured from capture, rather than from arrival. The original setpoint grid marks 0 and 25 ms as unavailable prefix, then hands off the 50 and 75 ms setpoints. Each handoff still requires a valid active lease and independent checks.'],
 ['Replace chunk 0 with its successor','At this endpoint chunk 1, captured 50 ms after chunk 0, is available and directly replaces it. Chunk 0’s evidence age is 90 ms; its 400 ms buffered tail is discarded from the active queue, shown dimmed in the original setpoint grid.'],
 ['Inspect the old chunk’s still-valid evidence','Continue inspecting chunk 0’s capture-relative evidence after it has been replaced. The original renewal lane is retained for context. This cursor does not put the old chunk back into force.'],
 ['Reach the first evidence horizon','At this endpoint the tracked-drift bound reaches grasp tolerance. The old chunk is already replaced; its now-expired remainder is explicitly withheld in the original grid. Its remaining buffered setpoints cannot extend that evidence horizon or create permission.'],
 ['Inspect the supply-only tail','The first chunk has 400 ms of capture-relative supply, but its evidence expired earlier. The hatched tail remains buffer, not permission. This inspection does not simulate continued execution of the old chunk.']
 ].map(([label,description],i)=>({label,get description(){return root.dataset.lessonCommittedIndex===String(i)?description:'Next endpoint, not yet completed: '+description;}}));
 function frame(t){const valid=t<D.horizon,first=t<D.latency?'unavailable':t<D.replacement?'in force':'replaced',age=t,error=D.initial+.5*D.slip*(age/1000)**2*1000;
  cursor.setAttribute('x1',X(t));cursor.setAttribute('x2',X(t));used.setAttribute('x2',X(Math.min(D.replacement,Math.max(D.latency,t))));supply.setAttribute('x2',X(t));supply.setAttribute('stroke',valid?'#B87333':'#A51C30');
  const counts={unavailable:0,queued:0,consumed:0,replaced:0,withheld:0};ticks.forEach(({time,mark,slash})=>{const status=time<D.latency?'unavailable':time<D.replacement&&time<=t?'consumed':t>=D.horizon&&time>=D.horizon?'withheld':t>=D.replacement&&time>=D.replacement?'replaced':'queued';counts[status]++;mark.dataset.status=status;mark.setAttribute('fill',status==='unavailable'?'#A0AEC0':status==='consumed'?'#1A4D3E':status==='withheld'?'#A51C30':'#B87333');mark.setAttribute('opacity',status==='replaced'?'.22':status==='queued'?'.55':'1');slash.setAttribute('display',status==='withheld'?'':'none');});
  handoff.style.display=t>=D.latency&&t<D.replacement?'':'none';handoff.setAttribute('cx',X(Math.min(75,Math.max(50,Math.floor(t/D.step)*D.step))));root.dataset.setpointCounts=JSON.stringify(counts);
  root.dataset.time=String(t);root.dataset.firstChunkState=first;root.dataset.firstEvidenceValid=String(valid);root.dataset.chunkExecutedMs=String(Math.min(D.period,Math.max(0,t-D.latency)));
  readout.textContent=`Chunk 0 capture age ${t.toFixed(1)} ms · ${first} · evidence ${valid?'within horizon':'expired'} · first-chunk execution ${Math.min(D.period,Math.max(0,t-D.latency)).toFixed(1)} ms · grid: ${counts.unavailable} unavailable, ${counts.consumed} handed off, ${counts.replaced} replaced, ${counts.withheld} withheld`;
  equation.textContent=`Chunk 0 error bound = e₀ + ½ a_slip t² = ${error.toFixed(2)} mm; tolerance ${D.tolerance.toFixed(1)} mm. Replacement ${D.replacement.toFixed(0)} ms < evidence horizon ${D.horizon.toFixed(1)} ms < supply ${D.span.toFixed(0)} ms. Lease validity is a separate requirement.`;
 }
 function render(i,{stage,details}){root.dataset.lessonCommittedIndex=String(i);mount(stage,details);frame(times[i]);follow(X(times[i]),i<2?1320:i<4?1650:1970);}
 function transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>{frame(times[from]+(times[to]-times[from])*p);root.dataset.animationProgress=String(p);});}

 return window.MLSysOriginalPlayer.create({root,outer:context.outer,title:"Why does buffered supply outlast permission?",scope:"Original direct-replacement timing strip. Exact chapter schedule and tracked-mug bound; no extra admission delay is charged. After replacement, the cursor inspects chunk 0 evidence and supply, not continued execution.",steps,render,transition});
};})();
