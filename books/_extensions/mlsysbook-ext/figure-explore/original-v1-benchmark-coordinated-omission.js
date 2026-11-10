/* Coordinated omission: one time cursor across the original two-policy timeline.
 * Visual layer (cursor, issued dots, growing traces) follows the interpolated
 * cursor; counts in the figure and ledger describe only the committed step. */
(()=>{'use strict';
const NAME='original-v1-benchmark-coordinated-omission',NS='http://www.w3.org/2000/svg';
// Generated and checked by original-v1-benchmark-coordinated-omission.calc.py from the figure's TikZ.
// Times are TikZ cm on the schematic axis; x0/x1/sy/head are drawing coordinates in the verified SVG.
const D={"pause":[4.0,8.4],"axis":[2.9,12.8],"scale":0.99585,"ox":0.547002,"oy":161.102681,"cursors":[2.9,4.0,4.3,7.9,8.4,9.1,9.6,10.3,10.9,10.9],"intervals":[{"policy":0,"arrival":4.3,"response":9.1,"x0":121.9309,"x1":253.0468,"sy":22.7815,"head":251.479829},{"policy":0,"arrival":5.5,"response":9.7,"x0":155.8055,"x1":269.9843,"sy":36.8959,"head":268.417081},{"policy":0,"arrival":6.7,"response":10.3,"x0":189.6801,"x1":286.9218,"sy":51.0103,"head":285.354751},{"policy":0,"arrival":7.9,"response":10.9,"x0":223.5547,"x1":303.8593,"sy":65.1247,"head":302.292003},{"policy":1,"arrival":4.3,"response":9.1,"x0":121.9309,"x1":253.0468,"sy":106.0565,"head":251.479829},{"policy":1,"arrival":9.6,"response":10.9,"x0":271.5437,"x1":303.8593,"sy":106.0565,"head":302.292003}]};
const GREEN='#008f45',RED='#d9534f',CURSOR='#1f407a',INK='#17212b',POLICY=['Open loop','Closed loop'];
const sx=t=>D.scale*t*72/2.54+D.ox,PAUSE=[sx(D.pause[0]),sx(D.pause[1])];
const svg=(tag,a={},text)=>{const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))n.setAttribute(k,v);if(text!=null)n.textContent=text;return n;};
const inPause=t=>D.pause[0]<=t&&t<D.pause[1];
// Same equations as the calc script: counts of drawn intervals at cursor t.
function ledger(t){
 const L={cursor:t,server:inPause(t)?'paused':'running',sent:[0,0],sentInPause:[0,0],outstanding:[0,0],recorded:[0,0],recordedFromPause:[0,0]};
 for(const iv of D.intervals){const k=iv.policy;if(iv.arrival>t)continue;L.sent[k]++;L.sentInPause[k]+=inPause(iv.arrival);
  if(iv.response<=t){L.recorded[k]++;L.recordedFromPause[k]+=inPause(iv.arrival);}else L.outstanding[k]++;}
 return L;
}
const ledgers=D.cursors.map(ledger),F=ledgers[ledgers.length-1];
const n=(v,word)=>v+' '+word+(v===1?'':'s');
// Narration derives its numbers from the committed ledgers; none are typed here.
const states=[
 ['Two arrival policies, one server',()=>'Both rows share one schematic time axis and the same shaded server pause. The cursor sits before the pause and neither policy has sent a request. Faded intervals are the ones the drawing shows later.'],
 ['The server pauses',()=>'The server stops responding where the shaded region begins, as during a garbage-collection or reallocation stall. Nothing has been sent yet, so nothing is waiting.'],
 ['Both policies send a request',L=>`Each policy issues its first request at the same moment inside the pause. The paused server cannot answer, so each policy has ${n(L.outstanding[0],'request')} outstanding.`],
 ['Independent arrivals keep coming',L=>`Open-loop arrivals do not wait for the server: ${n(L.sentInPause[0]-ledgers[2].sentInPause[0],'more request')} arrive during the pause, so ${L.outstanding[0]} are outstanding. The closed-loop client is blocked on its first response and sends nothing; its offered load during the pause stays at ${n(L.sentInPause[1],'request')}.`],
 ['The server resumes',L=>`The pause ends, but in the drawing no response has returned yet. The open-loop stream still has ${L.outstanding[0]} requests outstanding; the closed-loop client still has ${L.outstanding[1]}.`],
 ['First responses return',L=>`The first request of each policy completes, and each records one sample whose interval spans the pause. The open loop still has ${L.outstanding[0]} requests outstanding; the closed loop has ${L.outstanding[1]}.`],
 ['The closed-loop client sends again',L=>`Only after its response does the closed-loop client issue its next request. That request arrives after the pause, so its interval is short. The open loop still has ${L.outstanding[0]} requests outstanding.`],
 ['The open-loop backlog drains',L=>`The open loop records ${n(L.recorded[0]-ledgers[6].recorded[0],'more sample')}, each for a request that arrived during the pause. ${n(L.outstanding[0],'open-loop request')} and the closed-loop second request remain outstanding.`],
 ['The last responses return',L=>`Nothing is outstanding. The open loop recorded ${n(L.recorded[0],'sample')}, all from arrivals during the pause. The closed loop recorded ${n(L.recorded[1],'sample')}: ${L.recordedFromPause[1]} from the pause and ${L.recorded[1]-L.recordedFromPause[1]} after it.`],
 ['Compare what each harness recorded',L=>`Same server, same pause, different workloads. The independent stream offered ${n(L.sentInPause[0],'request')} during the pause and recorded every delayed one; the waiting client offered ${L.sentInPause[1]}, so its sample omits the delays that ${L.sentInPause[0]-L.sentInPause[1]} more arrivals would have experienced. The closed-loop result remains valid for a population of clients that wait this way; it does not estimate latency for independent arrivals. The harness must reproduce the deployment arrival policy or state why it differs.`]
];
const cues=['','','client blocked: sends nothing','client blocked: sends nothing','client still waiting','','','','',`sent during the pause: ${F.sentInPause[0]} vs ${F.sentInPause[1]}`];

window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons[NAME]=async(root,{outer,verifiedArtwork})=>{
 if(!verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified coordinated-omission diagram and shared player required');
 if(verifiedArtwork.getAttribute('viewBox')!=='0 0 362.984 186.224')throw Error('Coordinated-omission artwork geometry changed; review the mappings');
 root.classList.add(NAME);
 const art=verifiedArtwork.cloneNode(true);art.removeAttribute('width');art.removeAttribute('height');
 art.setAttribute('role','img');art.setAttribute('aria-label','Original timeline of one server pause under open-loop and closed-loop arrival policies, with a moving time cursor; requests appear as the cursor reaches their arrival.');
 // Bind the six drawn intervals (dot, line, arrowhead) in document order; verify against the calc export.
 const greens=[...art.querySelectorAll('path')].filter(p=>/56\.077576%/.test((p.getAttribute('fill')||'')+(p.getAttribute('stroke')||'')));
 const parts=D.intervals.map((iv,i)=>{const [dot,line,head]=greens.slice(3*i,3*i+3),m=/matrix\([^,]+, 0, 0, [^,]+, ([\d.]+),/.exec(head?.getAttribute('transform')||'');
  if(!line||!m||Math.abs(Number(m[1])-iv.head)>1e-6)throw Error('Interval '+i+' does not match the reviewed artwork');return{dot,line,head};});
 // Overlay: pause outline, outstanding traces, cursor, committed counts, and a cue inside the pause.
 const overlay=svg('g',{'data-co-overlay':'true'}),values=svg('g',{'data-mechanism-values':'true'});
 const pauseBox=svg('rect',{x:PAUSE[0],y:.6,width:PAUSE[1]-PAUSE[0],height:130.6,fill:'none',stroke:RED,'stroke-width':1.3});
 const traces=D.intervals.map(iv=>svg('line',{y1:iv.sy,y2:iv.sy,stroke:GREEN,'stroke-width':2.2,'stroke-linecap':'round'}));
 const cursor=svg('path',{fill:'none',stroke:CURSOR,'stroke-width':.9,'stroke-dasharray':'2.5 2','data-co-cursor':'true'}),tick=svg('circle',{cy:159.7,r:2.2,fill:CURSOR,'data-motion-object':'true'});
 overlay.append(pauseBox,...traces,cursor,tick);
 const counter=(y,label)=>{const t=svg('text',{x:70.5,y,'text-anchor':'end','font-size':6.4,fill:INK,'font-family':'sans-serif',class:'co-desktop'});t.append(label,svg('tspan',{'font-weight':'700'}));values.append(t);return t.lastChild;};
 const counts=[[counter(64,'outstanding '),counter(73,'recorded ')],[counter(125,'outstanding '),counter(134,'recorded ')]];
 const cue=svg('text',{x:(PAUSE[0]+PAUSE[1])/2,y:124,'text-anchor':'middle','font-size':6.4,'font-weight':'700',fill:INK,'font-family':'sans-serif'});values.append(cue);
 // Narrow screens crop to the timeline; these labels replace the cropped row titles.
 const mobile=svg('g',{class:'co-mobile','aria-hidden':'true'});
 [['Open loop',12],['Closed loop',97]].forEach(([label,y])=>{mobile.append(svg('rect',{x:81,y:y-8.4,width:label.length*4.6+4,height:11,rx:1.5,fill:'#fff','fill-opacity':.92}),svg('text',{x:83,y,'font-size':8.2,'font-weight':'700',fill:INK,'font-family':'sans-serif'},label));});
 mobile.append(svg('text',{x:313,y:168,'text-anchor':'end','font-size':7.5,fill:INK,'font-family':'sans-serif'},'time'));
 art.append(overlay,values,mobile);
 const source=document.createElement('div');source.className='co-source';source.append(art);
 // Text ledger: the same committed counts, readable at any width and in both themes.
 const ledgerBox=document.createElement('div'),caption=document.createElement('p'),table=document.createElement('table');ledgerBox.className='co-ledger-box';caption.className='co-ledger-note';table.className='co-ledger';ledgerBox.append(caption,table);
 const head=document.createElement('tr'),body=document.createElement('tbody');
 ['','Open loop','Closed loop'].forEach(h=>{const th=document.createElement('th');th.scope='col';th.textContent=h;head.append(th);});
 const thead=document.createElement('thead');thead.append(head);table.append(thead,body);
 const rows=['Sent during the pause','Outstanding now','Recorded samples','Recorded, arrived in the pause'].map(label=>{const tr=document.createElement('tr'),th=document.createElement('th');th.scope='row';th.textContent=label;tr.append(th);const cells=[0,1].map(()=>tr.appendChild(document.createElement('td')));body.append(tr);return cells;});
 const narrow=window.matchMedia('(max-width:700px)');
 function layout(){art.setAttribute('viewBox',narrow.matches?'80 0 235 171':'0 0 362.984 186.224');cue.setAttribute('font-size',narrow.matches?7.4:6.4);}
 narrow.addEventListener('change',layout);layout();
 // Visual state follows x (interpolated); nothing here writes a count.
 function paint(t){
  const x=sx(t);cursor.setAttribute('d',`M${x},0V74.6M${x},86.7V137.4M${x},148.6V159.7`);// gaps keep the two note texts legible
  tick.setAttribute('cx',x);
  pauseBox.style.visibility=inPause(t)?'visible':'hidden';
  D.intervals.forEach((iv,i)=>{const p=parts[i],issued=iv.arrival<=t+1e-9,done=iv.response<=t+1e-9;
   p.dot.setAttribute('opacity',issued?1:.16);p.line.setAttribute('opacity',done?1:issued?.28:.16);p.head.setAttribute('opacity',done?1:.16);
   const tr=traces[i];tr.style.visibility=issued&&!done?'visible':'hidden';tr.setAttribute('x1',iv.x0);tr.setAttribute('x2',Math.max(iv.x0,Math.min(x,iv.x1)));});
 }
 // Committed values: figure counters, cue, ledger table, and data attributes.
 function commit(i){
  const L=ledgers[i];
  counts.forEach(([o,r],k)=>{o.textContent=L.outstanding[k];r.textContent=L.recorded[k];});
  cue.textContent=cues[i];
  caption.textContent=`Step ${i+1}: counts of drawn requests; server ${L.server}. Schematic, not measurements.`;
  [0,1].forEach(k=>{rows[0][k].textContent=L.sentInPause[k];rows[1][k].textContent=L.outstanding[k];
   rows[2][k].textContent=L.recorded[k];rows[3][k].textContent=L.recordedFromPause[k];});
  table.classList.toggle('co-final',i===states.length-1);
  root.dataset.ledger=JSON.stringify(L);root.dataset.lessonState=String(i);
 }
 let committed=0,mounted=false;
 const steps=states.map(([label,describe],i)=>({label,get description(){return committed===i?describe(ledgers[i]):'In progress: '+label.toLowerCase()+'. The counts below still describe the last completed step.';}}));
 return window.MLSysOriginalPlayer.create({root,outer,layout:'compact',intro:`Step the time cursor across the shaded server pause to compare ${F.sentInPause[0]} open-loop arrivals against ${F.sentInPause[1]} closed-loop arrival during the stall.`,title:'Why does a closed-loop client record fewer delayed requests?',
  scope:'The original schematic timeline: one server pause and the drawn request intervals under two arrival policies. Every count is a count of drawn requests, not a measurement. Horizontal position is schematic time with no units, and animation speed represents no duration. Response order follows the drawing, not a queueing model.',steps,
  render(i,{stage,details}){committed=i;if(!mounted){stage.append(source);(details||stage).append(ledgerBox);mounted=true;}paint(D.cursors[i]);commit(i);delete root.dataset.pending;},
  transition({from,to,onFrame}){root.dataset.pending=String(to);const a=D.cursors[from],b=D.cursors[to];onFrame(p=>{paint(a+(b-a)*p);root.dataset.animationProgress=String(p);});}
 });
};})();
