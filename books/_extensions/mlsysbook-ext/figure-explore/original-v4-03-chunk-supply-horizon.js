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
 // Keep the endpoint in view when Expand or rotation changes the artwork scale.
 // Scrolling alone does not trigger this observer, so manual inspection persists.
 let focus=null,resizeFrame=0,disposed=false;
 function recenter(){if(!focus||disposed)return;if(canvas.clientWidth>=520){canvas.scrollLeft=0;canvas.scrollTop=0;return;}const scale=svg.getBoundingClientRect().width/width;canvas.scrollLeft=Math.max(0,focus[0]*scale-canvas.clientWidth/2);canvas.scrollTop=Math.max(0,focus[1]*scale-canvas.clientHeight/2);}
 function follow(x,y){focus=[x,y];recenter();}
 const cameraResize=typeof ResizeObserver==='function'?new ResizeObserver(()=>{if(!disposed&&!resizeFrame)resizeFrame=requestAnimationFrame(()=>{resizeFrame=0;recenter();});}):null;
 cameraResize?.observe(canvas);cameraResize?.observe(svg);
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
 ['Capture chunk 0 evidence','At t = 0 ms, the dashed teal cursor marks chunk 0 capture while the tracked mug error starts at e₀ = 3.00 mm. Sixteen capture-relative setpoints span 400 ms at 25 ms spacing, before inference completes.'],
 ['Make chunk 0 available','At t = 40 ms, inference finishes and chunk 0 enters force. Because the 0 ms and 25 ms setpoints elapsed during the 40 ms evaluation, the setpoint lane dims both as an unavailable prefix while slip drift reaches 3.40 mm.'],
 ['Execute within the replacement interval','At t = 65 ms, the dark green bar tracks active execution between 40 ms and 90 ms as the copper dot hands off the 50 ms setpoint (followed by 75 ms). Evidence age is 65 ms from capture, with a 4.06 mm error bound.'],
 ['Replace chunk 0 with its successor','At t = 90 ms (one 50 ms renewal period plus 40 ms inference), chunk 1 arrives from the t = 50 ms capture and replaces chunk 0 at 5.03 mm drift, dimming the 12 remaining setpoints in chunk 0.'],
 ['Inspect the old chunk’s still-valid evidence','Sweeping the cursor to t = 200 ms inspects chunk 0 after replacement: its slip error bound reaches 13.00 mm, still inside the 15.0 mm grasp tolerance while newer chunks command the arm.'],
 ['Reach the first evidence horizon','At t = 219.1 ms, tracked slip drift e₀ + ½ a_slip t² reaches the 15.00 mm grasp tolerance (red dashed line), so the 7 remaining setpoints from 225 ms to 375 ms are crossed out in red as expired.'],
 ['Inspect the supply-only tail','At t = 400 ms, chunk 0 reaches the end of its H = 16 setpoint span with a 43.00 mm drift bound; the hatched tail beyond 219.1 ms is buffered setpoint supply for jitter, never permission to execute.']
 ].map(([label,description],i)=>({label,get description(){return root.dataset.lessonCommittedIndex===String(i)?description:'Next endpoint, not yet completed: '+description;}}));
 function frame(t){const valid=t<D.horizon,first=t<D.latency?'unavailable':t<D.replacement?'in force':'replaced',age=t,error=D.initial+.5*D.slip*(age/1000)**2*1000;
  cursor.setAttribute('x1',X(t));cursor.setAttribute('x2',X(t));used.setAttribute('x2',X(Math.min(D.replacement,Math.max(D.latency,t))));supply.setAttribute('x2',X(t));supply.setAttribute('stroke',valid?'#B87333':'#A51C30');
  const counts={unavailable:0,queued:0,consumed:0,replaced:0,withheld:0};ticks.forEach(({time,mark,slash})=>{const status=time<D.latency?'unavailable':time<D.replacement&&time<=t?'consumed':t>=D.horizon&&time>=D.horizon?'withheld':t>=D.replacement&&time>=D.replacement?'replaced':'queued';counts[status]++;mark.dataset.status=status;mark.setAttribute('fill',status==='unavailable'?'#A0AEC0':status==='consumed'?'#1A4D3E':status==='withheld'?'#A51C30':'#B87333');mark.setAttribute('opacity',status==='replaced'?'.22':status==='queued'?'.55':'1');slash.setAttribute('display',status==='withheld'?'':'none');});
  handoff.style.display=counts.consumed>0&&t<D.replacement?'':'none';handoff.setAttribute('cx',X(Math.min(75,Math.max(50,Math.floor(t/D.step)*D.step))));root.dataset.setpointCounts=JSON.stringify(counts);
  root.dataset.time=String(t);root.dataset.firstChunkState=first;root.dataset.firstEvidenceValid=String(valid);root.dataset.chunkExecutedMs=String(Math.min(D.period,Math.max(0,t-D.latency)));
  readout.textContent=`Chunk 0 capture age ${t.toFixed(1)} ms · ${first} · evidence ${valid?'within horizon':'expired'} · first-chunk execution ${Math.min(D.period,Math.max(0,t-D.latency)).toFixed(1)} ms · grid: ${counts.unavailable} unavailable, ${counts.consumed} handed off, ${counts.replaced} replaced, ${counts.withheld} withheld`;
  equation.textContent=`Chunk 0 error bound = e₀ + ½ a_slip t² = ${error.toFixed(2)} mm; tolerance ${D.tolerance.toFixed(1)} mm. Replacement ${D.replacement.toFixed(0)} ms < evidence horizon ${D.horizon.toFixed(1)} ms < supply ${D.span.toFixed(0)} ms. Lease validity is a separate requirement.`;
 }
 function render(i,{stage,details}){root.dataset.lessonCommittedIndex=String(i);mount(stage,details);frame(times[i]);follow(X(times[i]),i<2?1320:i<4?1650:1970);}
 function transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>{frame(times[from]+(times[to]-times[from])*p);root.dataset.animationProgress=String(p);});}

 const player=window.MLSysOriginalPlayer.create({root,outer:context.outer,layout:'compact',intro:'Trace one 400 ms action chunk from capture through its 40 ms inference, 90 ms replacement, and 219.1 ms evidence horizon.',title:"Why does buffered supply outlast permission?",scope:"Direct-replacement conveyor schedule: 50 ms renewal period (20 Hz), 40 ms p99 inference latency, and H = 16 setpoints spaced 25 ms apart (400 ms span), with no extra admission delay. Tracked mug error grows from e₀ = 3.00 mm under residual belt slip a_slip = 0.5 m/s² toward the r_tol = 15.0 mm grasp tolerance. After replacement at 90 ms, the cursor inspects chunk 0 evidence and buffered supply rather than continued execution.",steps,render,transition});
 const destroy=player.destroy;player.destroy=()=>{disposed=true;cameraResize?.disconnect();cancelAnimationFrame(resizeFrame);destroy();};return player;
};})();
