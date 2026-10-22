/* Canonical timing lanes with a shared cursor and independently bounded timers. */
(()=>{'use strict';
const NAME='original-v1-benchmark-async-timing',NS='http://www.w3.org/2000/svg';
// Generated and checked by the prototype's calculations.py; these are drawing coordinates.
const G={"start":57.476574,"call":99.921895,"deviceStart":119.730493,"deviceStop":295.171933,"hostStop":320.636781,"deviceMid":207.451213};
const svg=(tag,a={})=>{const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))n.setAttribute(k,v);return n;};
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons[NAME]=async(root,{outer,verifiedArtwork})=>{
 if(!verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified timing diagram and shared player required');
 if(verifiedArtwork.getAttribute('viewBox')!=='0 0 358.388 133.764')throw Error('Timing artwork geometry changed; review the mappings');
 root.classList.add(NAME);
 const art=verifiedArtwork.cloneNode(true);art.classList.add('async-original');art.setAttribute('aria-label','Original host and accelerator timing diagram with a shared moving time cursor and three independently bounded timer traces');
 const source=document.createElement('div');source.className='async-source';source.tabIndex=0;source.setAttribute('aria-label','Aligned host and accelerator lanes with call, device-event, and synchronized host timing intervals.');source.append(art);
 const overlay=svg('g',{'data-async-overlay':'true'}),cursor=svg('line',{y1:2,y2:120,stroke:'#1f407a','stroke-width':.8,'stroke-dasharray':'2 2','pointer-events':'none'}),dot=svg('circle',{cy:2,r:1.8,fill:'#1f407a','data-motion-object':'true'});
 const rails=[svg('line',{y1:27.5,y2:27.5,stroke:'#008f45'}),svg('line',{y1:64.5,y2:64.5,stroke:'#e67917'}),svg('line',{y1:80.72,y2:80.72,stroke:'#008f45'}),svg('line',{y1:94.87,y2:94.87,stroke:'#e67917'}),svg('line',{y1:117.5,y2:117.5,stroke:'#1f407a'})];
 rails.forEach(n=>{n.setAttribute('stroke-width','1.6');n.setAttribute('stroke-linecap','round');});overlay.append(...rails,cursor,dot);art.append(overlay);
 // The narrow view reflows labels, retaining the same horizontal time boundaries.
 const mobile=svg('svg',{viewBox:'0 0 358.388 330',role:'img','aria-label':'Responsive version of the same two-lane timeline and three timing intervals'});mobile.classList.add('async-mobile');
 const text=(x,y,value,extra={})=>{const n=svg('text',{x,y,fill:'#17212b','font-size':14,...extra});n.textContent=value;mobile.append(n);return n;};
 const line=(x1,y1,x2,y2,color='#666',extra={})=>{const n=svg('line',{x1,y1,x2,y2,stroke:color,'stroke-width':1,...extra});mobile.append(n);return n;};
 const block=(x,y,w,h,fill,stroke)=>mobile.append(svg('rect',{x,y,width:w,height:h,fill,stroke,'stroke-width':1}));
 line(G.start,4,340,4);mobile.append(svg('path',{d:'M336,1L340,4L336,7',fill:'none',stroke:'#666'}));text(340,21,'time',{'text-anchor':'end','font-size':12});
 text(10,21,'Host');text(10,113,'Accelerator stream');
 line(G.start,56,340,56);line(G.start,148,340,148);
 block(G.start,36,G.call-G.start,40,'#e8fdc5','#008f45');text((G.start+G.call)/2,60,'enqueue',{'text-anchor':'middle','font-size':10});
 block(G.call,36,G.hostStop-G.call,40,'#f3f3f3','#777');text((G.call+G.hostStop)/2,53,'host waits',{'text-anchor':'middle'});text((G.call+G.hostStop)/2,69,'for completion',{'text-anchor':'middle'});
 block(G.deviceStart,128,G.deviceStop-G.deviceStart,40,'#fce3cb','#e67917');text((G.deviceStart+G.deviceStop)/2,153,'device work',{'text-anchor':'middle'});
 const down=svg('path',{d:`M${G.call},77V96H${G.deviceStart}V126`,fill:'none',stroke:'#666','stroke-dasharray':'3 3'}),up=svg('path',{d:`M${G.deviceStop},127V96H${G.hostStop}V78`,fill:'none',stroke:'#666','stroke-dasharray':'3 3'});mobile.append(down,up);
 mobile.append(svg('path',{d:`M${G.deviceStart-3},122L${G.deviceStart},126L${G.deviceStart+3},122M${G.hostStop-3},82L${G.hostStop},78L${G.hostStop+3},82`,fill:'none',stroke:'#666'}));
 const mobileStatuses=[text(340,204,'ready',{'text-anchor':'end'}),text(340,249,'not started',{'text-anchor':'end'}),text(340,294,'ready',{'text-anchor':'end'})];
 text(10,204,'Call timer');text(10,249,'Device events');text(10,294,'Synchronized host');
 const mobileRails=[];[[G.start,G.call,217,'#008f45'],[G.deviceStart,G.deviceStop,262,'#e67917'],[G.start,G.hostStop,307,'#1f407a']].forEach(([a,b,y,c])=>{line(a,y,b,y,c,{'stroke-opacity':.3,'stroke-width':2});mobileRails.push(line(a,y,a,y,c,{'stroke-width':3}));line(a,y-3,a,y+3,c);line(b,y-3,b,y+3,c);});
 const mobileDot=svg('circle',{cy:4,r:2,fill:'#1f407a'});mobile.append(mobileDot);
 const mobileCursor=svg('path',{fill:'none',stroke:'#1f407a','stroke-dasharray':'2 3','stroke-width':.8});mobile.append(mobileCursor);
 source.append(mobile);
 const readout=document.createElement('p');readout.className='async-readout';readout.setAttribute('aria-live','off');
 const states=[
 ['Set the timing boundaries','Both host timers are ready to start at the enqueue boundary. The device-event timer has not started.'],
 ['The host call returns','Enqueueing is complete. The call timer stops, but the accelerator has not reached the start event. The synchronized host timer keeps running while the host waits.'],
 ['The stream reaches the start event','The device-event interval starts here, after the host call returned. The host is still waiting at the completion barrier.'],
 ['The accelerator executes the work','Work advances inside the same accelerator block. The device-event and synchronized host intervals grow together; the short call interval remains frozen.'],
 ['The stream reaches the stop event','The bracketed device work is complete, so the device-event interval stops. The synchronized host timer continues until completion reaches the host.'],
 ['The host observes completion','The completion barrier returns and the synchronized host timer stops. The three traces have different endpoints because they answer different timing questions.']
 ];
 const positions=[G.start,G.call,G.deviceStart,G.deviceMid,G.deviceStop,G.hostStop];let committed=0,mounted=false;
 const steps=states.map(([label,description],i)=>({label,get description(){return committed===i?description:'In progress: '+label.toLowerCase()+'. The endpoint is not committed until this action finishes.';}}));
 function paint(x,initial=false){
  cursor.setAttribute('x1',x);cursor.setAttribute('x2',x);dot.setAttribute('cx',x);
  const starts=[G.start,G.deviceStart,G.start,G.deviceStart,G.start],ends=[G.call,G.deviceStop,G.call,G.deviceStop,G.hostStop];
  rails.forEach((n,i)=>{n.setAttribute('x1',starts[i]);n.setAttribute('x2',Math.max(starts[i],Math.min(x,ends[i])));n.style.visibility=x<=starts[i]?'hidden':'visible';});
  const call=initial?'ready':x<G.call?'running':'stopped',device=x<G.deviceStart?'not started':x<G.deviceStop?'running':'stopped',host=initial?'ready':x<G.hostStop?'running':'stopped';
  mobileStatuses[0].textContent=call;mobileStatuses[1].textContent=device;mobileStatuses[2].textContent=host;
  mobileDot.setAttribute('cx',x);mobileCursor.setAttribute('d',`M${x},6V35M${x},77V127M${x},169V185M${x},210V224M${x},255V269M${x},300V315`);
  mobileRails.forEach((n,i)=>{const a=[G.start,G.deviceStart,G.start][i],b=[G.call,G.deviceStop,G.hostStop][i];n.setAttribute('x2',Math.max(a,Math.min(x,b)));});
  readout.textContent=`Call timer: ${call} · Device events: ${device} · Synchronized host: ${host}`;
  root.dataset.timingState=JSON.stringify({committed,call,device,host,x});
  source.scrollLeft=Math.max(0,x*art.getBoundingClientRect().width/358.388-source.clientWidth/2);
 }
 const player=window.MLSysOriginalPlayer.create({root,outer,title:'Why can a host call finish before the accelerator?',scope:'The original schematic timeline, not measured milliseconds. One cursor shows time in both lanes. Strong traces grow only inside their declared timing boundaries; animation speed does not represent hardware speed. Earlier device work has completed before the host timer starts.',steps,
 render(i,{stage}){committed=i;if(!mounted){stage.append(source,readout);mounted=true;}paint(positions[i],i===0);root.dataset.lessonState=String(i);delete root.dataset.pending;},
 transition({from,to,onFrame}){root.dataset.pending=String(to);onFrame(p=>{paint(positions[from]+(positions[to]-positions[from])*p,from===0&&p===0);root.dataset.animationProgress=String(p);});}
 });return player;
};})();
