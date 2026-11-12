/* Original RNN unfolding; the same three hidden states live, wait, and are freed differently in inference and BPTT. */
(()=>{'use strict';
const NAME='original-v1-rnn-state-lifetime',NS='http://www.w3.org/2000/svg';
// Counts come from original-v1-rnn-state-lifetime.calc.py (RNNCompute's Scenarios.ElmanRnnStep in MLSysIM);
// `calc.py --check` asserts this block. The browser performs no arithmetic on lesson values.
// BEGIN calc.py output
const D={"hidden":128,"hidden_text":"128","input":100,"input_text":"100","recurrent_macs":16384,"recurrent_text":"16,384","input_macs":12800,"input_macs_text":"12,800","per_step":29184,"per_step_text":"29,184","steps_drawn":3,"inference_peak":128,"inference_peak_text":"128","training_peak":384,"training_peak_text":"384","input_ahead_text":"38,400","total_text":"87,552","chapter":{"T":1000,"T_text":"1,000","inference_values":128,"training_values":128000,"inference_bytes":512,"training_bytes":512000,"inference_values_text":"128","training_values_text":"128,000","inference_bytes_text":"512","training_bytes_text":"512,000"},"states":[{"mode":"inference","phase":"forward","nodes":["next","blocked","blocked"],"inputs":["pending","pending","pending"],"active":null,"grad":null,"retained":0,"retainedValues":0,"recurrentSteps":0,"recurrentMacs":0,"inputSteps":0,"inputMacs":0,"totalMacs":0,"whhReads":0,"peakValues":128,"retainedText":"0 values","recurrentText":"0","inputText":"0","totalText":"0","peakSoFar":0,"peakSoFarText":"0"},{"mode":"inference","phase":"forward","nodes":["live","next","blocked"],"inputs":["current","pending","pending"],"active":0,"grad":null,"retained":1,"retainedValues":128,"recurrentSteps":1,"recurrentMacs":16384,"inputSteps":1,"inputMacs":12800,"totalMacs":29184,"whhReads":1,"peakValues":128,"retainedText":"1 × 128 = 128 values","recurrentText":"1 × 16,384 = 16,384","inputText":"1 × 12,800 = 12,800","totalText":"29,184","peakSoFar":128,"peakSoFarText":"128"},{"mode":"inference","phase":"forward","nodes":["freed","live","next"],"inputs":["discarded","current","pending"],"active":1,"grad":null,"retained":1,"retainedValues":128,"recurrentSteps":2,"recurrentMacs":32768,"inputSteps":2,"inputMacs":25600,"totalMacs":58368,"whhReads":2,"peakValues":128,"retainedText":"1 × 128 = 128 values","recurrentText":"2 × 16,384 = 32,768","inputText":"2 × 12,800 = 25,600","totalText":"58,368","peakSoFar":128,"peakSoFarText":"128"},{"mode":"inference","phase":"forward","nodes":["freed","freed","live"],"inputs":["discarded","discarded","current"],"active":2,"grad":null,"retained":1,"retainedValues":128,"recurrentSteps":3,"recurrentMacs":49152,"inputSteps":3,"inputMacs":38400,"totalMacs":87552,"whhReads":3,"peakValues":128,"retainedText":"1 × 128 = 128 values","recurrentText":"3 × 16,384 = 49,152","inputText":"3 × 12,800 = 38,400","totalText":"87,552","peakSoFar":128,"peakSoFarText":"128"},{"mode":"training","phase":"forward","nodes":["next","blocked","blocked"],"inputs":["present","present","present"],"active":null,"grad":null,"retained":0,"retainedValues":0,"recurrentSteps":0,"recurrentMacs":0,"inputSteps":3,"inputMacs":38400,"totalMacs":38400,"whhReads":0,"peakValues":384,"retainedText":"0 values","recurrentText":"0","inputText":"3 × 12,800 = 38,400","totalText":"38,400","peakSoFar":0,"peakSoFarText":"0"},{"mode":"training","phase":"forward","nodes":["live","next","blocked"],"inputs":["present","present","present"],"active":0,"grad":null,"retained":1,"retainedValues":128,"recurrentSteps":1,"recurrentMacs":16384,"inputSteps":3,"inputMacs":38400,"totalMacs":54784,"whhReads":1,"peakValues":384,"retainedText":"1 × 128 = 128 values","recurrentText":"1 × 16,384 = 16,384","inputText":"3 × 12,800 = 38,400","totalText":"54,784","peakSoFar":128,"peakSoFarText":"128"},{"mode":"training","phase":"forward","nodes":["live","live","next"],"inputs":["present","present","present"],"active":1,"grad":null,"retained":2,"retainedValues":256,"recurrentSteps":2,"recurrentMacs":32768,"inputSteps":3,"inputMacs":38400,"totalMacs":71168,"whhReads":2,"peakValues":384,"retainedText":"2 × 128 = 256 values","recurrentText":"2 × 16,384 = 32,768","inputText":"3 × 12,800 = 38,400","totalText":"71,168","peakSoFar":256,"peakSoFarText":"256"},{"mode":"training","phase":"forward","nodes":["live","live","live"],"inputs":["present","present","present"],"active":2,"grad":null,"retained":3,"retainedValues":384,"recurrentSteps":3,"recurrentMacs":49152,"inputSteps":3,"inputMacs":38400,"totalMacs":87552,"whhReads":3,"peakValues":384,"retainedText":"3 × 128 = 384 values","recurrentText":"3 × 16,384 = 49,152","inputText":"3 × 12,800 = 38,400","totalText":"87,552","peakSoFar":384,"peakSoFarText":"384"},{"mode":"training","phase":"backward","nodes":["live","live","freed"],"inputs":["present","present","present"],"active":2,"grad":2,"retained":2,"retainedValues":256,"recurrentSteps":3,"recurrentMacs":49152,"inputSteps":3,"inputMacs":38400,"totalMacs":87552,"whhReads":3,"peakValues":384,"retainedText":"2 × 128 = 256 values","recurrentText":"3 × 16,384 = 49,152","inputText":"3 × 12,800 = 38,400","totalText":"87,552","peakSoFar":384,"peakSoFarText":"384"},{"mode":"training","phase":"backward","nodes":["live","freed","freed"],"inputs":["present","present","present"],"active":1,"grad":1,"retained":1,"retainedValues":128,"recurrentSteps":3,"recurrentMacs":49152,"inputSteps":3,"inputMacs":38400,"totalMacs":87552,"whhReads":3,"peakValues":384,"retainedText":"1 × 128 = 128 values","recurrentText":"3 × 16,384 = 49,152","inputText":"3 × 12,800 = 38,400","totalText":"87,552","peakSoFar":384,"peakSoFarText":"384"},{"mode":"training","phase":"backward","nodes":["freed","freed","freed"],"inputs":["present","present","present"],"active":0,"grad":0,"retained":0,"retainedValues":0,"recurrentSteps":3,"recurrentMacs":49152,"inputSteps":3,"inputMacs":38400,"totalMacs":87552,"whhReads":3,"peakValues":384,"retainedText":"0 values","recurrentText":"3 × 16,384 = 49,152","inputText":"3 × 12,800 = 38,400","totalText":"87,552","peakSoFar":384,"peakSoFarText":"384"},{"mode":"compare","phase":"compare","nodes":["idle","idle","idle"],"inputs":["present","present","present"],"active":null,"grad":null,"retained":0,"retainedValues":0,"recurrentSteps":3,"recurrentMacs":49152,"inputSteps":3,"inputMacs":38400,"totalMacs":87552,"whhReads":3,"peakValues":null,"retainedText":"0 values","recurrentText":"3 × 16,384 = 49,152","inputText":"3 × 12,800 = 38,400","totalText":"87,552"}]};
// END calc.py output
// Drawing coordinates in the verified SVG (viewBox 0 0 617.479 200.976): unrolled hidden-node centers,
// their radius, the output/input rows, and the white label plates the arrows pass through.
const HX=[235.551,394.16,552.766],HY=104.867,HR=19.83,YY=17.066,XY=183.887,SR=17;
const plate=(x,y,w,h)=>({x,y,w,h});
const WHX=HX.map(c=>plate(c-13.34,141.125,26.68,16.09)),WYH=HX.map(c=>plate(c-13.34,50.719,26.68,17.66));
// W_hh plates: into h_{t-1} (from the unfold arrow), t-1 -> t, t -> t+1, and leaving h_{t+1}.
const WHH=[plate(136.898,99.375,21.98,10.98),plate(301.309,96.82,27.09,16.09),plate(459.918,96.82,27.09,16.09),plate(580.012,99.375,21.98,10.98)];
const INTO=[110.04,255.38,413.98];  // where the horizontal arrow into each hidden node starts
const WIDE='0 -1 617.479 202.976',NARROW='198 -1 419.479 202.976';
const ORANGE='#e67817',ORANGE_DARK='#a34700',BLUE='#006395',GRAY='#5f6b73';
const SUB=['t−1','t','t+1'],UNI=['ₜ₋₁','ₜ','ₜ₊₁'],H=UNI.map(s=>'h'+s),X=UNI.map(s=>'x'+s);
const svg=(tag,a={},text)=>{const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))n.setAttribute(k,v);if(text!=null)n.textContent=text;return n;};
// Rendered width when attached; a conservative estimate while the panel is still detached.
const measure=(t,size)=>{let w=0;try{w=t.getComputedTextLength();}catch(e){}return w>0?w:t.textContent.length*size*.56;};
const html=(tag,cl,text)=>{const n=document.createElement(tag);if(cl)n.className=cl;if(text!=null)n.textContent=text;return n;};
// "h" with an HTML subscript, for the readable panel below the drawing.
const hsub=(base,k)=>{const s=html('span','rnn-sym',base);s.append(html('sub','',SUB[k]));return s;};

// Axis-aligned arrow segments with the label plate removed, clipped to fraction p of the path.
function segments(a,b,skip,p=1){
 const horiz=a[1]===b[1],A=horiz?a[0]:a[1],B=horiz?b[0]:b[1],end=A+(B-A)*p,lo=Math.min(A,end),hi=Math.max(A,end),out=[];
 const parts=skip?[[lo,Math.min(hi,skip[0])],[Math.max(lo,skip[1]),hi]]:[[lo,hi]];
 for(const[s,e]of parts)if(e>s+.01)out.push(horiz?[[s,a[1]],[e,a[1]]]:[[a[0],s],[a[0],e]]);
 return out;
}
const span=(pl,horiz)=>horiz?[pl.x,pl.x+pl.w]:[pl.y,pl.y+pl.h];
// Arrows the committed state names as read in its step. kind: 'fwd' (forward read) or 'back' (gradient).
function arrowsFor(s){
 const out=[];
 if(s.phase==='forward'&&s.active!=null){
  const k=s.active;out.push({a:[INTO[k],HY],b:[HX[k]-HR,HY],skip:span(WHH[k],true),kind:'fwd'});
  if(s.mode==='inference')out.push({a:[HX[k],XY-SR],b:[HX[k],HY+HR],skip:span(WHX[k],false),kind:'fwd'});
 }else if(s.mode==='training'&&s.phase==='forward'&&s.active==null){
  HX.forEach((c,k)=>out.push({a:[c,XY-SR],b:[c,HY+HR],skip:span(WHX[k],false),kind:'fwd'}));
 }else if(s.phase==='backward'){
  const k=s.active;out.push({a:[HX[k],YY+SR],b:[HX[k],HY-HR],skip:span(WYH[k],false),kind:'back'});
  if(k<2)out.push({a:[HX[k+1]-HR,HY],b:[HX[k]+HR,HY],skip:span(WHH[k+1],true),kind:'back'});
 }
 return out;
}
function platesFor(s,i){
 if(i===0)return{shared:[...WHX,...WYH,...WHH],read:[]};
 if(s.phase==='forward'&&s.active!=null)return{shared:[],read:s.mode==='inference'?[WHH[s.active],WHX[s.active]]:[WHH[s.active]]};
 if(s.mode==='training'&&s.phase==='forward')return{shared:[],read:WHX};
 if(s.phase==='backward')return{shared:[],read:s.active<2?[WYH[s.active],WHH[s.active+1]]:[WYH[s.active]]};
 return{shared:[],read:[]};
}

// One explanation per committed state; every count is a calc.py string.
function lesson(){
 const S=D.states,C=D.chapter;
 const text=[
  ['Inference: shared weights, no state yet',`Streaming inference starts here. The three matrices exist once each: W_hx (${D.input_text} × ${D.hidden_text}), W_hh (${D.hidden_text} × ${D.hidden_text}), and W_yh. The figure repeats their labels at every step, not their storage. No hidden state exists yet, and no input has arrived.`],
  [`Inference: step t−1 writes ${H[0]}`,`${X[0]} arrives. The step reads the initial state (zeros) through W_hh and ${X[0]} through W_hx: ${D.recurrent_text} + ${D.input_macs_text} = ${D.per_step_text} MACs. It writes ${H[0]}, the only state kept: ${D.hidden_text} values. ${H[1]} cannot start until ${X[1]} arrives, and ${H[2]} needs ${H[1]}.`],
  [`Inference: step t reads ${H[0]}, then frees it`,`Step t reads ${X[1]} through W_hx and ${H[0]} through the same W_hh copy, then writes ${H[1]}. That read was the last use of ${H[0]}, so the runtime frees or overwrites it; ${X[0]} was already discarded. Still one state is kept: ${D.hidden_text} values.`],
  [`Inference: step t+1, still one state`,`Step t+1 reads ${X[2]} through W_hx and ${H[1]} through W_hh, writes ${H[2]}, then ${H[1]} is freed. After three steps the kept state is still ${D.hidden_text} values, while W_hh has been read three times from one copy: ${S[3].recurrentText} recurrent MACs, executed in order.`],
  ['Training: inputs project ahead; the recurrence waits',`BPTT starts over on the same three steps with the whole sequence known. The input projections do not depend on any state, so all three may run together: ${S[4].inputText} MACs. Each W_hh product still needs the state before it, so ${H[0]} comes first and ${H[1]}, ${H[2]} wait.`],
  [`Training: forward step t−1 keeps ${H[0]}`,`Step t−1 adds its recurrent product (${D.recurrent_text} MACs) to the precomputed input projection and writes ${H[0]}. Training keeps it for the backward pass: ${S[5].retainedText}.`],
  [`Training: forward step t keeps ${H[0]} too`,`Step t reads ${H[0]} and writes ${H[1]}. Inference would free ${H[0]} here; BPTT keeps it, because the backward pass will need it. Retained: ${S[6].retainedText}.`],
  [`Training: forward step t+1, every state kept`,`Step t+1 writes ${H[2]}. All three states are retained: ${S[7].retainedText}, the forward-pass peak. The forward work equals inference's (${S[7].totalText} MACs); only the retained memory differs.`],
  [`Backward through step t+1, free ${H[2]}`,`The loss gradient enters ${H[2]} from ${'y'+UNI[2]} through W_yh. The backward step reads the states saved for it; afterward nothing needs ${H[2]}, so it is freed. ${H[1]} stays: step t's backward has not run. Retained: ${S[8].retainedText}.`],
  [`Backward through step t, free ${H[1]}`,`The gradient ∂L/∂h moves back through W_hh, a Jacobian product that reads W_hh again (backward work is not counted here), and joins the gradient from ${'y'+UNI[1]}. Step t's backward was the last reader of ${H[1]}, so it is freed. Retained: ${S[9].retainedText}.`],
  [`Backward through step t−1, free ${H[0]}`,`The gradient reaches ${H[0]}; its backward step runs and ${H[0]} is freed. Because the weights are shared, every step's contribution was added into one gradient per matrix, not one per step. Retained: 0 values.`],
  ['Compare: O(H) versus O(T·H)',`Inference kept one state (${D.inference_peak_text} values) at every step; BPTT peaked at ${D.training_peak_text} values for the toy T = ${D.steps_drawn}. At the scenario's ${C.T_text}-step sequence, inference still keeps ${C.inference_values_text} values while BPTT keeps ${C.training_values_text}. Both run ${C.T_text} recurrent steps in order. Recomputing saved states instead trades extra work for memory.`]
 ];
 if(!Array.isArray(S)||S.length!==text.length)throw Error('RNN lesson data missing or stale');
 return S.map((s,i)=>({...s,label:text[i][0],description:text[i][1]}));
}

window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons[NAME]=async(root,{outer,verifiedArtwork})=>{
 if(!verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified RNN unfolding diagram and shared player required');
 if(verifiedArtwork.getAttribute('viewBox')!=='0 0 617.479 200.976')throw Error('RNN artwork geometry changed; review the mappings');
 root.classList.add(NAME);
 const states=lesson();
 const art=verifiedArtwork.cloneNode(true);art.removeAttribute('width');art.removeAttribute('height');art.classList.add('rnn-original');
 art.setAttribute('role','img');art.setAttribute('aria-label','Original RNN unfolding with the status of each unrolled hidden state; the same statuses are listed below the diagram.');
 // Layers: read arrows under node marks, then plate outlines, badges, and the in-motion layer on top.
 const arrowLayer=svg('g',{class:'rnn-arrows'}),nodeLayer=svg('g',{class:'rnn-nodes'}),plateLayer=svg('g',{class:'rnn-plates'}),badgeLayer=svg('g',{class:'rnn-badges'}),motion=svg('g',{'aria-hidden':'true',class:'rnn-motion'});
 art.append(arrowLayer,nodeLayer,plateLayer,badgeLayer,motion);
 const source=html('div','rnn-source');source.tabIndex=0;source.setAttribute('aria-label','Original unrolled recurrent network; the status of each hidden state is also listed below.');source.append(art);
 // Readable panel: one slot per unrolled hidden state, the retained-memory ledger, and the work ledger.
 const panel=html('div','rnn-panel'),mode=html('p','rnn-mode'),slots=html('div','rnn-slots'),memory=html('p','rnn-memory'),work=html('p','rnn-work'),compare=html('div','rnn-compare');
 const slotNodes=SUB.map((_,k)=>{const c=html('div','rnn-slot');c.dataset.slot=String(k);const name=html('span','rnn-slot-name');name.append(hsub('h',k));c.append(name,html('span','rnn-slot-status'),html('span','rnn-slot-size'));slots.append(c);return c;});
 panel.append(mode,slots,memory,work,compare);
 const narrow=matchMedia('(max-width:600px)');let committed=0,mounted=false;
 const layout=()=>art.setAttribute('viewBox',narrow.matches?NARROW:WIDE);narrow.addEventListener?.('change',layout);layout();
 const steps=states.map((s,i)=>({label:s.label,get description(){return committed===i?s.description:'Pending: '+s.label+'. The retained states and counts below change only when this action completes.';}}));

 const veil=(cx,cy,r,opacity,dash)=>{nodeLayer.append(svg('circle',{cx,cy,r:r+.6,fill:'#fff','fill-opacity':opacity}));if(dash)nodeLayer.append(svg('circle',{cx,cy,r:r+1.6,fill:'none',stroke:dash,'stroke-width':1.1,'stroke-dasharray':'3 2'}));};
 function badge(k,kind,words,sub){
  const g=svg('g',{'data-badge':String(k),'data-kind':kind}),style={kept:['#d6eaf8',BLUE,'#003f5f'],freed:['#fff','#8a949b','#3d4850'],next:['#fce4cc',ORANGE_DARK,'#5c2c00'],waits:['#fff','#8a949b','#3d4850']}[kind];
  const t=svg('text',{x:HX[k]-12,y:79.2,'text-anchor':'end','font-size':10,'font-family':'sans-serif',fill:style[2]},words);
  if(sub){t.append(svg('tspan',{'font-size':7.4,dy:2.2},sub));}
  const box=svg('rect',{y:68.7,height:13.6,rx:2.5,fill:style[0],stroke:style[1],'stroke-width':.8});g.append(box,t);badgeLayer.append(g);
  const w=measure(t,10)+7;box.setAttribute('x',HX[k]-8.5-w);box.setAttribute('width',w);
 }
 function chip(cx,label,color){const g=svg('g',{'data-chip':label}),t=svg('text',{x:cx,y:93.6,'text-anchor':'middle','font-size':8.8,'font-family':'sans-serif',fill:color},label),box=svg('rect',{y:84.6,height:11.8,rx:2,fill:'#fff',stroke:color,'stroke-width':.7});g.append(box,t);badgeLayer.append(g);const w=measure(t,8.8)+6;box.setAttribute('x',cx-w/2);box.setAttribute('width',w);}
 function line(layer,{a,b,skip,kind},p=1,width=1.8){for(const[[x1,y1],[x2,y2]]of segments(a,b,skip,p))layer.append(svg('line',{x1,y1,x2,y2,stroke:ORANGE,'stroke-width':width,'stroke-linecap':'round','stroke-dasharray':kind==='back'?'4 2.5':'none','data-arrow':kind}));}
 function outline(pl,color,dash,role){plateLayer.append(svg('rect',{'data-role':role||'shared',x:pl.x-1.4,y:pl.y-1.2,width:pl.w+2.8,height:pl.h+2.4,rx:2,fill:'none',stroke:color,'stroke-width':1.3,'stroke-dasharray':dash||'none'}));}

 function paint(i){
  const s=states[i];[arrowLayer,nodeLayer,plateLayer,badgeLayer].forEach(n=>n.replaceChildren());
  for(const a of arrowsFor(s))line(arrowLayer,a);
  const{shared,read}=platesFor(s,i);shared.forEach(pl=>outline(pl,BLUE,'3 2'));read.forEach(pl=>outline(pl,ORANGE,null,'read'));
  s.nodes.forEach((st,k)=>{
   const cx=HX[k];
   if(st==='live'){nodeLayer.append(svg('circle',{cx,cy:HY,r:HR+3.4,fill:'none',stroke:BLUE,'stroke-width':2.4,'data-node':'live'}));badge(k,'kept','kept');}
   else if(st==='freed'){veil(cx,HY,HR,.74,GRAY);badge(k,'freed','freed');}
   else if(st==='next'){veil(cx,HY,HR,.45,null);nodeLayer.append(svg('circle',{cx,cy:HY,r:HR+3.4,fill:'none',stroke:ORANGE_DARK,'stroke-width':1.5,'stroke-dasharray':'3 2'}));badge(k,'next','next');}
   else if(st==='blocked'){veil(cx,HY,HR,.6,GRAY);badge(k,'waits','waits for h',SUB[k-1]);}
   // An output exists only once its hidden state has been written.
   if(st==='next'||st==='blocked')veil(cx,YY,SR,.55,null);
  });
  s.inputs.forEach((st,k)=>{if(st==='pending')veil(HX[k],XY,SR,.62,GRAY);else if(st==='discarded')veil(HX[k],XY,SR,.6,null);});
  if(s.nodes[0]==='next'||(s.phase==='forward'&&s.active===0))chip(166,'initial state = zeros',GRAY);
  if(s.phase==='backward'&&s.active>0)chip(s.active===2?510:351.4,'← ∂L/∂h',ORANGE_DARK);
  // Panel: mode, slots, ledgers.
  mode.textContent={inference:'Inference (streaming)',training:s.phase==='backward'?'Training (BPTT), backward pass':'Training (BPTT), forward pass',compare:'Comparison'}[s.mode];
  mode.dataset.mode=s.mode;
  // Phones hide the SVG chips; the mode line repeats their cue.
  const cue=s.nodes[0]==='next'||(s.phase==='forward'&&s.active===0)?'initial state = zeros':s.phase==='backward'?'gradient flows ←':'';
  if(cue)mode.append(html('span','rnn-mode-cue',' · '+cue));
  const word={live:'kept',freed:'freed',next:'next',idle:''};
  s.nodes.forEach((st,k)=>{const c=slotNodes[k],status=c.children[1];c.dataset.status=st;status.replaceChildren();
   if(st==='blocked'){status.append('waits for ',hsub('h',k-1));}else status.textContent=word[st];
   c.children[2].textContent=st==='live'?D.hidden_text+' values':'';});
  slots.hidden=s.mode==='compare';
  if(s.mode==='compare'){
   const C=D.chapter;memory.textContent=`Retained hidden state: inference ${D.inference_peak_text} values at every step; BPTT peak ${D.training_peak_text} values at toy T = ${D.steps_drawn}.`;
   work.textContent=`Same forward work in both: ${D.total_text} MACs for three steps, three ordered forward reads of one W_hh copy (backward work not counted).`;
   compare.replaceChildren(
    row('Inference',1,`${C.inference_values_text} values (${C.inference_bytes_text} bytes in FP32) at any sequence length`),
    row('BPTT',3,`${D.training_peak_text} values at toy T = ${D.steps_drawn}; ${C.training_values_text} values (${C.training_bytes_text} bytes) at T = ${C.T_text}`));
   compare.hidden=false;
  }else{
   compare.hidden=true;compare.replaceChildren();
   memory.textContent=`Retained hidden state: ${s.retainedText} · peak so far ${s.peakSoFarText} values`;
   work.textContent=s.phase==='backward'
    ?`Forward MACs (backward work not counted): recurrent ${s.recurrentText} (in order) · input ${s.inputText} · W_hh: one copy, forward reads ${s.whhReads}×`
    :`MACs so far: recurrent ${s.recurrentText} (in order) · input ${s.inputText} · W_hh: one copy, read ${s.whhReads}×`;
  }
  root.dataset.ledger=JSON.stringify({mode:s.mode,phase:s.phase,nodes:s.nodes,inputs:s.inputs,retainedValues:s.retainedValues,peakSoFar:s.peakSoFar??null,recurrentMacs:s.recurrentMacs,inputMacs:s.inputMacs,totalMacs:s.totalMacs,whhReads:s.whhReads});
  root.dataset.endpoint=String(i);root.dataset.retained=String(s.retainedValues);
 }
 function row(name,count,note){const r=html('div','rnn-compare-row');r.dataset.regime=name.toLowerCase();const bar=html('span','rnn-compare-bar');for(let k=0;k<3;k++){const cell=html('span','rnn-compare-cell');cell.dataset.filled=String(k<count);bar.append(cell);}r.append(html('strong','',name),bar,html('span','rnn-compare-note',note));return r;}

 const detailReadout=html('p','rnn-detail-readout');
 const player=window.MLSysOriginalPlayer.create({root,outer,layout:'compact',intro:'Compare streaming inference, which keeps 1 hidden state of 128 values across 3 steps, against BPTT training, which retains all 384 values.',title:'Why does training keep every state that inference frees?',
  scope:`The figure's three drawn steps are treated as a toy complete sequence (T = ${D.steps_drawn}) entered from an initial state of zeros, with the caption's ${D.hidden_text}-wide hidden state and ${D.input_text}-wide input. Per-step MACs (${D.recurrent_text} recurrent + ${D.input_macs_text} input) come from the chapter's RNNCompute cell in MLSysIM; output projections and backward-pass work are not counted. Counts are hidden states held after each step completes, following the chapter's list h₁ … h_T; inputs and other saved tensors are not counted. During an inference update the old and new state briefly coexist. The ${D.chapter.T_text}-step comparison uses the same MLSysIM scenario. Motion is illustrative and does not encode time.`,
  steps,
  render(i,{stage,details}){committed=i;if(!mounted){stage.append(source);if(details)details.append(detailReadout,panel);mounted=true;}motion.replaceChildren();paint(i);detailReadout.textContent=`${mode.textContent} · ${memory.textContent} · ${work.textContent}`;delete root.dataset.pending;},
  transition({from,to,onFrame}){
   root.dataset.pending=String(to);motion.replaceChildren();
   // Read traces are events of the committed step, not retained state: clear them so only the moving read shows.
   arrowLayer.replaceChildren();plateLayer.querySelectorAll('[data-role=read]').forEach(n=>n.remove());
   const a=states[from],b=states[to],paths=arrowsFor(b),layers=paths.map(()=>svg('g'));layers.forEach(g=>motion.append(g));
   // Releases fade in after the read reaches its node; a newly kept node's ring appears last.
   const fades=[],rings=[];
   b.nodes.forEach((st,k)=>{
    if(st==='freed'&&a.nodes[k]==='live'&&a.mode===b.mode){const v=svg('circle',{cx:HX[k],cy:HY,r:HR+.6,fill:'#fff','fill-opacity':0});motion.append(v);fades.push(v);}
    if(st==='live'&&a.nodes[k]!=='live'){const r=svg('circle',{cx:HX[k],cy:HY,r:HR+3.4,fill:'none',stroke:BLUE,'stroke-width':2.4,opacity:0});motion.append(r);rings.push(r);}
   });
   onFrame(p=>{
    const reach=Math.min(1,p/.65);paths.forEach((path,j)=>{layers[j].replaceChildren();line(layers[j],path,reach,2.6);});
    const late=Math.max(0,(p-.65)/.35);fades.forEach(v=>v.setAttribute('fill-opacity',String(.74*late)));rings.forEach(r=>r.setAttribute('opacity',String(late)));
    root.dataset.animationProgress=String(p);
   });
  }});
 return player;
};})();
