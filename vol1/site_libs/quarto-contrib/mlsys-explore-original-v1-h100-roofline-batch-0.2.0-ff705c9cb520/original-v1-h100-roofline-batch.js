/* Original H100 roofline; the FP16 classifier-head marker walks the bandwidth slope as batch grows. */
(()=>{'use strict';
const NAME='original-v1-h100-roofline-batch',NS='http://www.w3.org/2000/svg';
// MLSysIM values (H100 peak/bandwidth, eq-batch-ai intensities, traffic bytes) and PNG pixel geometry,
// exported by original-v1-h100-roofline-batch.calc.py; `calc.py --check` asserts this block.
// The browser formats no lesson value; it only interpolates marker positions between endpoints.
// BEGIN calc.py output
const D={"art":{"width":3315,"height":2476,"sha1":"d4348c048b6e857332e172a12bf2b40445b174e9","frame":{"x0":373.0,"x1":3256.0,"y0":59.0,"y1":2138.0},"x_lim":[0.5,4000.0],"y_lim":[1.0,6000.0]},"h100":{"peak":989.0,"bw":3.35,"ridge":295.2238805970149,"ridge_int8":590.7462686567164,"text":{"peak":"989","bw":"3.35","ridge":"295.2","ridge_int8":"590.7"}},"head":{"inputs":2048,"outputs":1000,"limit":671.9160104986877,"limit_text":"671.9","printed_ai":30.545280992721633,"cross":527,"below":526,"below_ai_text":"295.0"},"ridge_xy":[2419.92,489.84],"slope_start_xy":[373.0,2014.73],"ceiling_end_xy":[3256.0,489.84],"states":[{"batch":32,"ai":30.545280992721633,"attainable":102.32669132561747,"pct":10.346480417150401,"bound":"bandwidth","mflop":131.072,"weight_mb":4.096,"act_mb":0.195072,"xy":[1692.2,1031.97],"text":{"batch":"32","ai":"30.5","attainable":"102.3","pct":"10.3","mflop":"131.1","weight_mb":"4.10","act_mb":"0.20","total_mb":"4.29"}},{"batch":1,"ai":0.9985139304394631,"attainable":3.3450216669722015,"pct":0.3382226154673611,"bound":"bandwidth","mflop":4.096,"weight_mb":4.096,"act_mb":0.006096,"xy":[594.88,1849.44],"text":{"batch":"1","ai":"1.0","attainable":"3.3","pct":"0.3","mflop":"4.1","weight_mb":"4.10","act_mb":"0.01","total_mb":"4.10"}},{"batch":256,"ai":185.37291817523533,"attainable":620.9992758870384,"pct":62.790624457738964,"bound":"bandwidth","mflop":1048.576,"weight_mb":4.096,"act_mb":1.560576,"xy":[2270.64,601.05],"text":{"batch":"256","ai":"185.4","attainable":"621.0","pct":"62.8","mflop":"1,048.6","weight_mb":"4.10","act_mb":"1.56","total_mb":"5.66"}},{"batch":512,"ai":290.5788876276958,"attainable":973.439273552781,"pct":98.42662017722759,"bound":"bandwidth","mflop":2097.152,"weight_mb":4.096,"act_mb":3.121152,"xy":[2414.83,493.63],"text":{"batch":"512","ai":"290.6","attainable":"973.4","pct":"98.4","mflop":"2,097.2","weight_mb":"4.10","act_mb":"3.12","total_mb":"7.22"}},{"batch":527,"ai":295.3499114466918,"attainable":989.0,"pct":100.0,"bound":"compute","mflop":2158.592,"weight_mb":4.096,"act_mb":3.212592,"xy":[2420.06,489.84],"text":{"batch":"527","ai":"295.3","attainable":"989.0","pct":"100.0","mflop":"2,158.6","weight_mb":"4.10","act_mb":"3.21","total_mb":"7.31"}},{"batch":1024,"ai":405.7052297939778,"attainable":989.0,"pct":100.0,"bound":"compute","mflop":4194.304,"weight_mb":4.096,"act_mb":6.242304,"xy":[2521.9,489.84],"text":{"batch":"1,024","ai":"405.7","attainable":"989.0","pct":"100.0","mflop":"4,194.3","weight_mb":"4.10","act_mb":"6.24","total_mb":"10.34"}}],"ledgers":[{"batch":32,"ai":"30.5","attainable":"102.3","pct":"10.3","bound":"bandwidth","weight_mb":"4.10","act_mb":"0.20"},{"batch":1,"ai":"1.0","attainable":"3.3","pct":"0.3","bound":"bandwidth","weight_mb":"4.10","act_mb":"0.01"},{"batch":256,"ai":"185.4","attainable":"621.0","pct":"62.8","bound":"bandwidth","weight_mb":"4.10","act_mb":"1.56"},{"batch":512,"ai":"290.6","attainable":"973.4","pct":"98.4","bound":"bandwidth","weight_mb":"4.10","act_mb":"3.12"},{"batch":527,"ai":"295.3","attainable":"989.0","pct":"100.0","bound":"compute","weight_mb":"4.10","act_mb":"3.21"},{"batch":1024,"ai":"405.7","attainable":"989.0","pct":"100.0","bound":"compute","weight_mb":"4.10","act_mb":"6.24"}],"expected_final":{"batch":1024,"ai":"405.7","attainable":"989.0","pct":"100.0","bound":"compute"}};
// END calc.py output
// Inline readout plate in PNG pixels: the empty plot area under the legend, above the slope.
const PANEL={x:420,y:868,w:940,h:340},R={dot:30,halo:46};
const svg=(tag,a={},text)=>{const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))n.setAttribute(k,v);if(text!=null)n.textContent=text;return n;};
const html=(tag,cl,text)=>{const n=document.createElement(tag);if(cl)n.className=cl;if(text!=null)n.textContent=text;return n;};

// The figure's own log axes: x 0.5..4000 FLOP/byte, y 1..6000 TFLOP/s, mapped onto the measured frame.
function place(ai){
 const{frame:f,x_lim:xl,y_lim:yl}=D.art,v=Math.min(D.h100.peak,D.h100.bw*ai),L=Math.log10;
 return[f.x0+(L(ai)-L(xl[0]))/(L(xl[1])-L(xl[0]))*(f.x1-f.x0),f.y1-(L(v)-L(yl[0]))/(L(yl[1])-L(yl[0]))*(f.y1-f.y0)];
}

// Ordered endpoints: one batch change per step. Text uses only calc.py strings.
function lesson(){
 const[s32,s1,s256,s512,sx,s1k]=D.states,h=D.h100.text,t=s=>s.text;
 return[
  [s32,'Printed point: batch 32',`The classifier maps ${D.head.inputs.toLocaleString('en-US')} pooled features to ${D.head.outputs.toLocaleString('en-US')} classes in FP16. At batch 32 it performs ${t(s32).mflop} MFLOP while moving ${t(s32).total_mb} MB, an intensity of ${t(s32).ai} FLOP/byte. The bandwidth slope caps it at ${t(s32).attainable} TFLOP/s, ${t(s32).pct}% of the ${h.peak} TFLOP/s FP16 ceiling. This is the filled marker in the printed figure.`],
  [s1,'Drop to batch 1',`With one image, each weight fetched from HBM serves one multiply-add. The ${t(s1).weight_mb} MB of weights is nearly all the traffic, so intensity falls to ${t(s1).ai} FLOP/byte and the bound to ${t(s1).attainable} TFLOP/s. The marker lands on GPT-2 XL decode at batch 1: both read every FP16 weight once per output. The dashed ring keeps the printed batch-32 position.`],
  [s256,'Raise the batch to 256',`Each fetched weight now serves 256 images. FLOPs grow with the batch while the weight bytes stay ${t(s256).weight_mb} MB, so intensity rises to ${t(s256).ai} FLOP/byte and the bound to ${t(s256).attainable} TFLOP/s (${t(s256).pct}% of peak). The point climbs the bandwidth slope; bandwidth still binds.`],
  [s512,'Batch 512 stays just below the ridge',`Intensity reaches ${t(s512).ai} FLOP/byte, not 512: activation traffic (${t(s512).act_mb} MB) also grows with the batch, so the approximation intensity ≈ batch no longer holds. The point is still left of the ${h.ridge} FLOP/byte FP16 ridge, bandwidth bound at ${t(s512).pct}% of peak.`],
  [sx,`Batch ${t(sx).batch} crosses the FP16 ridge`,`Batch ${t(sx).batch} is the smallest batch whose intensity, ${t(sx).ai} FLOP/byte, clears the ${h.ridge} FLOP/byte ridge (batch ${D.head.below} gives ${D.head.below_ai_text}). The binding term switches from the bandwidth slope to the ${h.peak} TFLOP/s FP16 ceiling: the layer is now compute bound.`],
  [s1k,'Batch 1,024: intensity rises, the bound does not',`Intensity grows to ${t(s1k).ai} FLOP/byte, but the bound stays at the ${h.peak} TFLOP/s ceiling. More batching buys no attainable throughput here, and even an unlimited batch only approaches ${D.head.limit_text} FLOP/byte for this layer. The cost is waiting: requests queue until a batch of this size forms. The roofline gives a bound, not a measured latency or per-request speed.`]
 ].map(([s,label,text])=>({s,label,text}));
}

window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons[NAME]=async(root,{outer,verifiedArtwork})=>{
 if(!verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified roofline plot and shared player required');
 if(verifiedArtwork.dataset.mlsysArtworkKind!=='png'||verifiedArtwork.getAttribute('viewBox')!==`0 0 ${D.art.width} ${D.art.height}`)throw Error('Roofline PNG geometry changed; review the mappings');
 root.classList.add(NAME);
 const L=lesson();let committed=0,mounted=false;
 const plot=verifiedArtwork.cloneNode(true);plot.removeAttribute('width');plot.removeAttribute('height');
 plot.setAttribute('role','img');plot.setAttribute('aria-label','Original H100 roofline with the ResNet-50 classifier head marker moving along the bandwidth slope and FP16 ceiling as batch size changes');
 const layer=svg('g',{'data-rb-overlay':'true'});plot.append(layer);
 // Binding term: the bandwidth slope up to the FP16 ridge, or the FP16 ceiling after it.
 const[rx,ry]=D.ridge_xy,[sx0,sy0]=D.slope_start_xy,[cx1]=D.ceiling_end_xy;
 const slope=svg('line',{x1:sx0,y1:sy0,x2:rx,y2:ry,stroke:'#006395','stroke-width':44,'stroke-opacity':.28,'stroke-linecap':'round','data-rb-binding':'bandwidth'});
 const ceiling=svg('line',{x1:rx,y1:ry,x2:cx1,y2:ry,stroke:'#e67817','stroke-width':44,'stroke-opacity':.3,'stroke-linecap':'round','data-rb-binding':'compute'});
 // The printed batch-32 dot stays visible but faded once the marker leaves it.
 const[gx,gy]=D.states[0].xy,fade=svg('circle',{cx:gx,cy:gy,r:R.dot+3,fill:'#fff','fill-opacity':.7}),ghost=svg('circle',{cx:gx,cy:gy,r:R.halo,fill:'none',stroke:'#555','stroke-width':6,'stroke-dasharray':'14 10'});
 const halo=svg('circle',{r:R.halo,fill:'#ffd34d',stroke:'#7a5a00','stroke-width':4,'data-mlsys-focus':''}),dot=svg('circle',{r:R.dot,fill:'#333','data-motion-object':'true'});
 // The inline readout plate is step-critical: the dock and the jump pill never cover it.
 const panel=svg('g',{'data-rb-panel':'true','data-mlsys-keep':''}),plate=svg('rect',{x:PANEL.x,y:PANEL.y,width:PANEL.w,height:PANEL.h,rx:18,fill:'#fff','fill-opacity':.94,stroke:'#7a5a00','stroke-width':5});
 // A swatch matching the moving marker ties the plate to it without a leader crossing printed labels.
 const swatch=[svg('circle',{cx:PANEL.x+66,cy:PANEL.y+62,r:30,fill:'#ffd34d',stroke:'#7a5a00','stroke-width':4}),svg('circle',{cx:PANEL.x+66,cy:PANEL.y+62,r:19,fill:'#333'})];
 const lines=[0,1,2,3].map(k=>svg('text',{x:PANEL.x+(k?36:116),y:PANEL.y+82+k*76,'font-size':k?56:58,'font-weight':k?400:700,fill:'#1d1d1d','font-family':'Helvetica Neue, Helvetica, Arial, sans-serif'}));
 panel.append(plate,...swatch,...lines);
 // The x-axis tick labels and title (below the measured frame) stay readable under the dock and pill.
 const axisKeep=svg('rect',{x:D.art.frame.x0,y:D.art.frame.y1,width:D.art.frame.x1-D.art.frame.x0,height:D.art.height-D.art.frame.y1,fill:'none','data-mlsys-keep':''});
 layer.append(slope,ceiling,fade,ghost,halo,dot,panel,axisKeep);
 const box=html('div','rb-plot');box.append(plot);
 // Readable values at every width: the same committed endpoint as the plot.
 const readout=html('div','rb-readout');readout.setAttribute('aria-live','off');
 const cell=(k)=>{const c=html('div','rb-cell'),v=html('span','rb-value');c.append(html('span','rb-key',k),v);readout.append(c);return v;};
 const vBatch=cell('Batch'),vAi=cell('Arithmetic intensity'),vAtt=cell('Attainable (roofline bound)'),vBound=cell('Binding term');
 const traffic=html('div','rb-traffic'),bar=html('div','rb-bar'),wSeg=html('div','rb-seg rb-weights'),aSeg=html('div','rb-seg rb-acts'),tNote=html('p','rb-traffic-note');
 bar.append(wSeg,aSeg);traffic.append(html('p','rb-traffic-title','Bytes moved per call (FP16): weights are fixed, activations grow with the batch'),bar,tNote);
 const maxMb=Math.max(...D.states.map(s=>s.weight_mb+s.act_mb));
 const small=window.matchMedia('(max-width: 700px)');
 function sizes(){const k=small.matches?2.3:1;dot.setAttribute('r',R.dot*k);halo.setAttribute('r',R.halo*k);halo.setAttribute('stroke-width',4*k);}
 function moveTo([x,y]){dot.setAttribute('cx',x);dot.setAttribute('cy',y);halo.setAttribute('cx',x);halo.setAttribute('cy',y);
  const away=Math.hypot(x-gx,y-gy)>R.dot;fade.style.visibility=ghost.style.visibility=away?'visible':'hidden';}
 function paintPanel(s,title){const t=s.text;lines[0].textContent=title;lines[1].textContent=`${t.ai} FLOP/byte`;lines[2].textContent=`${t.attainable} TFLOP/s (${t.pct}% of peak)`;lines[3].textContent=s.bound==='compute'?'compute bound':'bandwidth bound';}
 function commit(i){
  const s=L[i].s,t=s.text;committed=i;moveTo(s.xy);panel.classList.remove('is-pending');paintPanel(s,`Classifier · batch ${t.batch}`);
  slope.style.visibility=s.bound==='bandwidth'?'visible':'hidden';ceiling.style.visibility=s.bound==='compute'?'visible':'hidden';
  vBatch.textContent=t.batch;vAi.textContent=`${t.ai} FLOP/byte (ridge ${D.h100.text.ridge})`;vAtt.textContent=`${t.attainable} TFLOP/s · ${t.pct}% of ${D.h100.text.peak}`;
  vBound.textContent=s.bound==='compute'?'FP16 compute ceiling':'HBM bandwidth slope';vBound.dataset.bound=s.bound;
  wSeg.style.width=100*s.weight_mb/maxMb+'%';aSeg.style.width=100*s.act_mb/maxMb+'%';
  tNote.textContent=`Weights ${t.weight_mb} MB + activations ${t.act_mb} MB = ${t.total_mb} MB for ${t.mflop} MFLOP.`;
  root.dataset.ledger=JSON.stringify(D.ledgers[i]);root.dataset.batch=String(s.batch);root.dataset.bound=s.bound;delete root.dataset.pending;
 }
 // Mid-move the player shows the target step's own text under a "Next ·" heading; the plate and
 // readout keep the committed endpoint until arrival.
 const steps=L.map(({label,text})=>({label,description:text}));
 small.addEventListener?.('change',sizes);sizes();
 return window.MLSysOriginalPlayer.create({root,outer,title:'Where does batching move the classifier on the H100 roofline?',
  scope:`H100 values from MLSysIM: ${D.h100.text.bw} TB/s HBM bandwidth, ${D.h100.text.peak} TFLOP/s dense FP16 peak, FP16 ridge ${D.h100.text.ridge} FLOP/byte. ResNet-50 classifier head, ${D.head.inputs.toLocaleString('en-US')} inputs to ${D.head.outputs.toLocaleString('en-US')} outputs, FP16 weights and activations, one read of input and weights and one write of output per call (the chapter's batch-intensity equation and dense-layer notebook). Only this marker moves; every other printed point, both ceilings, and both ridges stay as printed. The INT8 ceiling is not used for this FP16 marker. Attainable throughput is the roofline bound, not a measured rate, and motion speed represents no time.`,
  steps,
  // Compact layout: the plot leads; the value cells and byte bar follow the narration.
  // Intro numbers come from the same calc.py states as the step labels.
  layout:'compact',intro:`Move the Classifier (FC head) marker from the batch-${D.states[0].text.batch} point to batch ${D.states[1].text.batch}, then to batch ${D.states[5].text.batch}, and watch it climb the bandwidth slope to the dense FP16 ceiling.`,
  render(i,{stage,details}){if(!mounted){stage.append(box);details.append(readout,traffic);mounted=true;}commit(i);},
  transition({from,to,onFrame}){
   root.dataset.pending=String(to);const a=Math.log(L[from].s.ai),b=Math.log(L[to].s.ai);
   // Committed values stay in the plate and readout; only the marker and the pending batch label move.
   paintPanel(L[from].s,`Batch ${L[from].s.text.batch} → ${L[to].s.text.batch}`);panel.classList.add('is-pending');
   onFrame(p=>moveTo(place(Math.exp(a+(b-a)*p))));
  }});
};})();
