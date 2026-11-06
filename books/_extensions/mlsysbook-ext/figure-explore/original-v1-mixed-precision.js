/* Original six-step FP16 cycle; one declared toy gradient pair follows the numbered arrows. */
(()=>{'use strict';
const NAME='original-v1-mixed-precision',NS='http://www.w3.org/2000/svg';
// Lesson values are IEEE binary16/bfloat16 results exported by original-v1-mixed-precision.calc.py
// (numpy.float16); `calc.py --check` asserts this block. The browser performs no rounding.
// BEGIN calc.py output
const D={"fp16":{"min_normal":6.103515625e-05,"min_normal_text":"2⁻¹⁴ ≈ 6.10 × 10⁻⁵","max":65504.0,"max_text":"65,504","min_subnormal":5.960464477539063e-08,"min_subnormal_text":"2⁻²⁴ ≈ 5.96 × 10⁻⁸"},"toy":{"w32":0.10000000149011612,"w32_text":"0.100000001","loss":0.75,"loss_text":"0.75","g_small":1.4901161193847656e-08,"g_small_text":"2⁻²⁶ ≈ 1.49 × 10⁻⁸","g_large":3.0,"g_large_text":"3","eta":0.5,"eta_text":"0.5"},"cast":{"w16":0.0999755859375,"w16_text":"0.0999755859375","w16_bits":"0x2E66","error":-2.4415552616119385e-05,"error_text":"−2.44 × 10⁻⁵","w32_text":"0.100000001","w16_token":"0.09998"},"trials":[{"exponent":0,"S":1.0,"S_plain":"1","S_pow":"1","S_text":"1 (no scaling)","scaled_loss":0.75,"scaled_loss_text":"0.75","exact_small":1.4901161193847656e-08,"exact_small_text":"2⁻²⁶ ≈ 1.49 × 10⁻⁸","exact_large":3.0,"exact_large_text":"3","small":0.0,"small_text":"0","small_class":"underflow","small_bits":"0x0000","small_ftz":0.0,"large":3.0,"large_text":"3","large_class":"normal","large_bits":"0x4200","finite":true},{"exponent":10,"S":1024.0,"S_plain":"1,024","S_pow":"2¹⁰","S_text":"2¹⁰ = 1,024","scaled_loss":768.0,"scaled_loss_text":"768","exact_small":1.52587890625e-05,"exact_small_text":"2⁻¹⁶ ≈ 1.53 × 10⁻⁵","exact_large":3072.0,"exact_large_text":"3,072","small":1.52587890625e-05,"small_text":"2⁻¹⁶ ≈ 1.53 × 10⁻⁵","small_class":"subnormal","small_bits":"0x0100","small_ftz":0.0,"large":3072.0,"large_text":"3,072","large_class":"normal","large_bits":"0x6A00","finite":true},{"exponent":15,"S":32768.0,"S_plain":"32,768","S_pow":"2¹⁵","S_text":"2¹⁵ = 32,768","scaled_loss":24576.0,"scaled_loss_text":"24,576","exact_small":0.00048828125,"exact_small_text":"2⁻¹¹ ≈ 4.88 × 10⁻⁴","exact_large":98304.0,"exact_large_text":"98,304","small":0.00048828125,"small_text":"2⁻¹¹ ≈ 4.88 × 10⁻⁴","small_class":"normal","small_bits":"0x1000","small_ftz":0.00048828125,"large":Infinity,"large_text":"inf","large_class":"overflow","large_bits":"0x7C00","finite":false},{"exponent":14,"S":16384.0,"S_plain":"16,384","S_pow":"2¹⁴","S_text":"2¹⁴ = 16,384","scaled_loss":12288.0,"scaled_loss_text":"12,288","exact_small":0.000244140625,"exact_small_text":"2⁻¹² ≈ 2.44 × 10⁻⁴","exact_large":49152.0,"exact_large_text":"49,152","small":0.000244140625,"small_text":"2⁻¹² ≈ 2.44 × 10⁻⁴","small_class":"normal","small_bits":"0x0C00","small_ftz":0.000244140625,"large":49152.0,"large_text":"49,152","large_class":"normal","large_bits":"0x7A00","finite":true}],"unscale_overflow":{"S_pow":"2¹⁵","S_plain":"32,768","small":1.4901161193847656e-08,"small_text":"2⁻²⁶ ≈ 1.49 × 10⁻⁸","large":"inf","large_text":"inf","finite":false},"unscaled":{"small":1.4901161193847656e-08,"small_text":"2⁻²⁶ ≈ 1.49 × 10⁻⁸","large":3.0,"large_text":"3"},"update":{"eta_text":"0.5","delta":7.450580596923828e-09,"delta_text":"2⁻²⁷ ≈ 7.45 × 10⁻⁹","w_new":0.09999999403953552,"w_new_text":"0.099999994","ulp32":7.450580596923828e-09,"ulp32_text":"2⁻²⁷ ≈ 7.45 × 10⁻⁹","ulp16":6.103515625e-05,"ulp16_text":"2⁻¹⁴ ≈ 6.10 × 10⁻⁵"},"bf16":{"small":1.4901161193847656e-08,"small_text":"2⁻²⁶ ≈ 1.49 × 10⁻⁸","large":3.0,"large_text":"3","w":0.10009765625,"w_text":"0.100097656","error_text":"9.77 × 10⁻⁵","loss":0.75,"loss_text":"0.75"}};
// END calc.py output
// Drawing coordinates in the verified SVG (viewBox 0 0 226.312 135.47): six boxes and the white step-label plates.
const BOX={grad32:[.47,.375,76.68,28.24],master:[148.48,.375,76.68,28.24],w16:[148.48,53.355,76.68,28.235],fwd:[148.48,106.336,76.68,28.234],sloss:[.47,106.336,76.68,28.234],sgrad:[.47,53.355,76.68,28.235]};
const PLATE={1:[174.06,34.67,25.52,8.79],2:[168.38,87.7,36.88,8.68],3:[98.63,116.04,28.37,8.82],4:[18.24,90.93,41.14,9.91],5:[8.65,37.89,60.32,10.03],6:[96.15,9.54,33.32,9.91]};
const ARROW={1:[[186.82,28.6],[186.82,53]],2:[[186.82,81.6],[186.82,106]],3:[[148.1,120.45],[77.5,120.45]],4:[[38.81,106],[38.81,81.96]],5:[[38.81,53],[38.81,28.98]],6:[[77.5,14.49],[148.1,14.49]]};
// The moving token swerves to this side of its arrow while passing the step-label plate (the phone view crops the outer tag columns).
const SIDE={1:[1,0],2:[[1,0],[-1,0]],3:[0,1],4:[1,0],5:[[-1,0],[1,0]],6:[0,-1]};
const NODE={master:'FP32 Master Weights',w16:'FP16 Weights',fwd:'Forward Pass (FP16 Loss)',sloss:'Scaled Loss (FP32)',sgrad:'Scaled FP16 Gradients',grad32:'FP32 Gradients'};
// Both views leave about 6 units (8+ px) of margin around the outlined boxes, so no highlight touches the frame.
const WIDE='-112 -8 450.6 151.5',NARROW='-8 -8 242.3 151.5',TAG_W=104;
const svg=(tag,a={},text)=>{const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))n.setAttribute(k,v);if(text!=null)n.textContent=text;return n;};
const html=(tag,cl,text)=>{const n=document.createElement(tag);if(cl)n.className=cl;if(text!=null)n.textContent=text;return n;};
const tight=s=>s.replace(/ × /g,'×'),short=s=>tight(s.includes(' ≈ ')?s.split(' ≈ ')[1]:s);

// Committed endpoints. Each state changes one causal relationship and lists every node value it implies.
function lesson(){
 const[t1,t10,t15,t14]=D.trials,T=D.toy,F=D.fp16,U=D.update;
 const pending={master:['FP32 · toy w',T.w32_text,'toy value',''],w16:['FP16 copy of w','—','not cast yet',''],fwd:['FP16 · toy loss L','—','not computed',''],sloss:['FP32 · S × L','—','not scaled',''],sgrad:['FP16 · S × g','—','not computed',''],grad32:['FP32 · g after ÷ S','—','not written','']};
 const cls=c=>({underflow:'bad',overflow:'bad',subnormal:'warn',normal:'ok'})[c];
 const scaled=t=>[`g: ${short(t.small_text)} ${t.small_class}`,`max: ${t.large_text} ${t.large_class==='overflow'?'(overflow)':'(finite)'}`,cls(t.small_class),t.large_class==='overflow'?'bad':''];
 const out=[];let tags={...pending},S=null,status=['',''],last={};
 const add=(label,description,change,extra={})=>{
  Object.assign(tags,change.tags||{});if('S'in change)S=change.S;if(change.status)status=change.status;Object.assign(last,change.ledger||{});
  out.push({label,description,tags:{...tags},S,bf16:!!change.bf16,status:[...status],active:extra.active||[],plate:extra.plate||null,marks:extra.marks||[],token:extra.token||null,ledger:{...last}});
 };
 const trial=t=>({tags:{sloss:['FP32 · S × L',`S·L = ${t.scaled_loss_text}`,`S = ${t.exponent?t.S_pow+' = '+t.S_plain:'1'}`,''],sgrad:pending.sgrad},S:t,status:['scaled loss',t.scaled_loss_text],ledger:{S:t.S,scaledLoss:t.scaled_loss,scaledSmall:null,scaledLarge:null,smallClass:null,largeClass:null,grad32Small:null,grad32Large:null}});
 const back=(t,status)=>{const[a,b,c,d]=scaled(t);return{tags:{sgrad:['FP16 · S × g',a,b,c,d]},status,ledger:{scaledSmall:t.small,scaledLarge:Number.isFinite(t.large)?t.large:'inf',smallClass:t.small_class,largeClass:t.large_class}};};
 add('Declare the toy values',`The FP32 master weight holds a toy w = ${T.w32_text}. This iteration's toy gradients are g = ${T.g_small_text} for the tracked weight and ${T.g_large_text} for the largest entry of the same gradient tensor. FP16's smallest subnormal is ${F.min_subnormal_text}, its smallest normal ${F.min_normal_text}, and its largest finite value ${F.max_text}.`,
  {ledger:{w32:T.w32,w16:null,loss:null,S:null,scaledLoss:null,scaledSmall:null,scaledLarge:null,smallClass:null,largeClass:null,grad32Small:null,grad32Large:null,update:'none'},status:['toy g values',`${short(T.g_small_text)} · ${T.g_large_text}`]},{active:['master']});
 add('Cast the master weight to FP16 (stage 1)',`Casting rounds w to the nearest binary16 value, ${D.cast.w16_text} (bits ${D.cast.w16_bits}), an error of ${D.cast.error_text}. The FP32 master copy keeps ${D.cast.w32_text}.`,
  {tags:{w16:['FP16 copy of w',D.cast.w16_text,`${D.cast.w16_bits} · error ${tight(D.cast.error_text)}`,'warn']},ledger:{w16:D.cast.w16}},{active:['w16'],plate:1,token:[1,D.cast.w16_token]});
 add('Run the forward pass in FP16 (stage 2)',`The forward pass reads the FP16 weights and produces a toy loss L = ${T.loss_text}. The gradients that follow from this loss are fixed by the toy setup; only how they are stored changes below.`,
  {tags:{fwd:['FP16 · toy loss L',`L = ${T.loss_text}`,'',''],},ledger:{loss:T.loss}},{active:['fwd'],plate:2,token:[2,'L = '+T.loss_text]});
 add('Scale by S = 1, no scaling (stage 3)',`With S = 1 the scaled loss is still ${t1.scaled_loss_text}. Backpropagation will produce the gradients unchanged.`,trial(t1),{active:['sloss'],plate:3,token:[3,'× 1']});
 add('Backprop at S = 1: g underflows (stage 4)',`The tracked gradient ${T.g_small_text} is below half of FP16's smallest subnormal, so it rounds to 0. The largest entry, ${t1.large_text}, is stored exactly. An update would leave the tracked weight unchanged: its gradient is lost, not merely small.`,
  back(t1,['g underflows','stored as 0']),{active:['sgrad'],plate:4,token:[4,'g·S']});
 add(`Compare: scale by S = ${t10.S_pow}`,`Rescale the same loss by the chapter's smaller example, S = ${t10.S_plain}. The scaled loss is ${t10.scaled_loss_text}. The S = 1 gradients are discarded; this compares scale factors on the same forward pass.`,trial(t10),{active:['sloss'],plate:3,token:[3,'× '+t10.S_pow]});
 add(`Backprop at S = ${t10.S_pow}: g is subnormal (stage 4)`,`S·g = ${t10.small_text} lands in FP16's subnormal band. IEEE binary16 keeps it, but hardware that flushes subnormals to zero would store 0. The largest entry becomes ${t10.large_text}.`,
  back(t10,['g subnormal','FTZ would give 0']),{active:['sgrad'],plate:4,token:[4,'g·S']});
 add(`Compare: scale by S = ${t15.S_pow}`,`Same forward pass, larger what-if factor: S = ${t15.S_plain}, the chapter's larger example. The scaled loss is ${t15.scaled_loss_text}.`,trial(t15),{active:['sloss'],plate:3,token:[3,'× '+t15.S_pow]});
 add(`Backprop at S = ${t15.S_pow}: the large entry overflows (stage 4)`,`S·g = ${t15.small_text} is a normal FP16 value, so the tracked gradient survives. But ${T.g_large_text} × ${t15.S_plain} = ${t15.exact_large_text} exceeds ${F.max_text} and becomes inf. One nonfinite entry is enough to invalidate the step.`,
  back(t15,['max → inf','overflow']),{active:['sgrad'],plate:4,token:[4,'g·S']});
 const O=D.unscale_overflow;
 add(`Copy & unscale at S = ${O.S_pow}: inf found (stage 5)`,`Copying to FP32 and dividing by S = ${O.S_plain} returns the tracked gradient to exactly ${O.small_text}, but inf ÷ S is still inf. A dynamic scaler checks these unscaled FP32 gradients for nonfinite values before the update runs.`,
  {tags:{grad32:['FP32 · g after ÷ S',`g: ${short(O.small_text)} (exact)`,'max: inf (nonfinite)','ok','bad']},status:['inf found','after ÷ '+O.S_pow],ledger:{grad32Small:O.small,grad32Large:O.large}},{active:['grad32'],plate:5,token:[5,'÷ '+O.S_pow]});
 add('Dynamic scaler: skip the update, halve S',`Because the unscaled gradients contain inf, the scaler skips step 6: the update does not run, so the nonfinite value never reaches the FP32 master weight, which stays ${T.w32_text}. S halves from ${t15.S_pow} to ${t14.S_text} for the next iteration. This is a scaler rule, not a numbered step of the figure.`,
  {tags:{grad32:['FP32 · g after ÷ S','max: inf','update skipped','bad','bad'],master:['FP32 · toy w',T.w32_text,'untouched (skipped)','']},S:t14,status:['update skipped',`next S = ${t14.S_pow}`],ledger:{S:t14.S,update:'skipped'}},{active:['master'],marks:[6]});
 add(`Next iteration: scale by S = ${t14.S_pow}`,`Toy assumption: the next iteration has the same weight, loss, and gradients. Scaling by S = ${t14.S_plain} gives a scaled loss of ${t14.scaled_loss_text}.`,
  {...trial(t14),tags:{...trial(t14).tags,grad32:pending.grad32,master:pending.master},ledger:{...trial(t14).ledger,update:'none'}},{active:['sloss'],plate:3,token:[3,'× '+t14.S_pow]});
 add(`Backprop at S = ${t14.S_pow}: both values fit (stage 4)`,`S·g = ${t14.small_text} stays above the ${F.min_normal_text} normal floor, and ${T.g_large_text} × ${t14.S_plain} = ${t14.large_text} stays below ${F.max_text}. Every scaled entry is finite.`,
  back(t14,['all finite','S fits both']),{active:['sgrad'],plate:4,token:[4,'g·S']});
 add('Copy to FP32 and unscale (stage 5)',`Dividing by S = ${t14.S_plain} returns the tracked gradient to exactly ${D.unscaled.small_text} and the largest entry to ${D.unscaled.large_text}. Because S is a power of two, scaling and unscaling only shift the exponent here.`,
  {tags:{grad32:['FP32 · g after ÷ S',`g: ${short(D.unscaled.small_text)} (exact)`,`max: ${D.unscaled.large_text}`,'ok']},status:['unscaled','÷ '+t14.S_pow],ledger:{grad32Small:D.unscaled.small,grad32Large:D.unscaled.large}},{active:['grad32'],plate:5,token:[5,'÷ '+t14.S_pow]});
 add('Update the FP32 master weight (stage 6)',`With a toy learning rate η = ${U.eta_text}, the change is −${U.delta_text}, exactly one FP32 step at this weight: w becomes ${U.w_new_text}. Near 0.1, adjacent FP16 values are ${U.ulp16_text} apart, so the same change applied to an FP16 copy would round away.`,
  {tags:{master:['FP32 · toy w',U.w_new_text,`moved −${U.delta_text.split(' ≈ ')[0]} (1 FP32 step)`,'ok']},status:['update applied','w − η·g'],ledger:{w32:U.w_new,update:'applied'}},{active:['master'],plate:6,token:[6,'− η·g']});
 const B=D.bf16;
 add('BF16 comparison: no scale path',`BF16 keeps FP32's 8-bit exponent, so the gradient ${B.small_text} and the entry ${B.large_text} are representable without scaling. Step 3 is dropped, and step 5 still copies the gradients to FP32 but has nothing to unscale. That is why BF16 usually omits loss scaling. The cost is mantissa precision: w = ${T.w32_text} becomes ${B.w_text} in BF16. The tags now show this BF16 run on the same toy values; the boxes keep the figure's FP16 labels. Framework implementations vary.`,
  {tags:{master:['FP32 · toy w',T.w32_text,'BF16 run: same toy w',''],w16:['BF16 copy of w',B.w_text,`error +${tight(B.error_text)}`,'warn'],fwd:['BF16 · toy loss L',`L = ${B.loss_text}`,'',''],
   sloss:['No loss scale','S not applied','BF16 drops step 3',''],sgrad:['BF16 · g (no S)',`g: ${short(B.small_text)} kept`,`max: ${B.large_text} (finite)`,'ok'],grad32:['FP32 · g (copy, no ÷ S)',`g: ${short(B.small_text)}`,`max: ${B.large_text}`,'ok']},
   status:['BF16: g kept','S not needed'],bf16:true,ledger:{bf16Small:B.small,bf16Large:B.large,bf16W:B.w,bf16Loss:B.loss}},{marks:[3],active:[]});
 return out;
}

window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons[NAME]=async(root,{outer,verifiedArtwork})=>{
 if(!verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified mixed-precision diagram and shared player required');
 if(verifiedArtwork.getAttribute('viewBox')!=='0 0 226.312 135.47')throw Error('Mixed-precision artwork geometry changed; review the mappings');
 if(!D.trials)throw Error('Mixed-precision lesson data missing');
 root.classList.add(NAME);
 const states=lesson();
 const art=verifiedArtwork.cloneNode(true);art.removeAttribute('width');art.removeAttribute('height');art.classList.add('mp-original');
 art.setAttribute('role','img');art.setAttribute('aria-label','Original FP16 mixed-precision cycle with the stored value and data type beside each box');
 // The center panel (loss scale S and the step's outcome, e.g. "max → inf overflow") is step-critical: keep it uncovered.
 const marks=svg('g',{'data-mp-marks':'true'}),tagLayer=svg('g',{class:'mp-tags'}),panel=svg('g',{class:'mp-panel','data-mlsys-keep':''}),motion=svg('g',{'aria-hidden':'true'});
 art.append(marks,tagLayer,panel,motion);
 // Value tags sit outside the original canvas, level with their boxes; the drawing itself is not moved or relabeled.
 const tagNodes={};
 for(const[key,[x,y,w,h]]of Object.entries(BOX)){
  const left=x<100,tx=left?-110:232.3,g=svg('g',{'data-node':key});
  g.append(svg('rect',{x:tx,y:y-2.5,width:TAG_W,height:h+5,rx:2,fill:'#fff',stroke:'#8a949b','stroke-width':.5}));
  g.append(svg('line',{x1:left?tx+TAG_W:x+w,y1:y+h/2,x2:left?x:tx,y2:y+h/2,stroke:'#8a949b','stroke-width':.5,'stroke-dasharray':'1.5 1.5'}));
  const lines=[svg('text',{x:tx+3.5,y:y+6.2,'font-size':6.4,'font-weight':700,fill:'#17212b'}),svg('text',{x:tx+3.5,y:y+16,'font-size':7.2,'font-weight':600,fill:'#17212b'}),svg('text',{x:tx+3.5,y:y+25.4,'font-size':6.6,fill:'#3d4850'})];
  g.append(...lines);tagLayer.append(g);tagNodes[key]=lines;
 }
 panel.append(svg('rect',{x:80.5,y:29,width:65,height:78,rx:3,fill:'#fff',stroke:'#8a949b','stroke-width':.5}));
 const panelText=[[43,6.6,700,'#17212b'],[55,7.4,600,'#17212b'],[79,6.8,700,'#17212b'],[90,6.6,400,'#17212b']].map(([y,s,w,f])=>{const n=svg('text',{x:113,y,'font-size':s,'font-weight':w,fill:f,'text-anchor':'middle'});panel.append(n);return n;});
 const source=html('div','mp-source');source.tabIndex=0;source.setAttribute('aria-label','Original six-step mixed-precision cycle; the current values are also listed below the diagram.');source.append(art);
 // Narrow screens keep the drawing full width and move the same node values into a 2 × 3 grid with the drawing's layout.
 const grid=html('div','mp-grid'),cards={};
 for(const key of['grad32','master','sgrad','w16','sloss','fwd']){const c=html('div','mp-card');c.dataset.node=key;c.dataset.kind=key==='master'||key==='grad32'?'fp32':key==='w16'||key==='fwd'?'fp16':'scaled';c.append(html('span','mp-card-name',NODE[key]),html('span','mp-card-dtype'),html('span','mp-card-value'),html('span','mp-card-note'));grid.append(c);cards[key]=c;}
 const readout=html('p','mp-readout');readout.setAttribute('aria-live','off');
 const narrow=matchMedia('(max-width:600px)');let committed=0,mounted=false;
 const layout=()=>art.setAttribute('viewBox',narrow.matches?NARROW:WIDE);narrow.addEventListener?.('change',()=>{layout();if(mounted)paint(committed);});layout();
 // Mid-move the player shows the target step's real explanation under a "Next ·" heading;
 // stored values (tags, cards, panel) still describe the committed step until arrival.
 const steps=states.map(s=>({label:s.label,description:s.description}));
 const tone={ok:'#0b6b37',warn:'#a34700',bad:'#b11d33','':'#3d4850'};
 function fit(n,max){n.removeAttribute('textLength');n.removeAttribute('lengthAdjust');try{if(n.getComputedTextLength()>max){n.setAttribute('textLength',max);n.setAttribute('lengthAdjust','spacingAndGlyphs');}}catch(e){}}
 function paint(i){
  const s=states[i];marks.replaceChildren();
  for(const key of s.active){const[x,y,w,h]=BOX[key];marks.append(svg('rect',{x:x-1.6,y:y-1.6,width:w+3.2,height:h+3.2,rx:2,fill:'none',stroke:'#cc5500','stroke-width':1.6,'data-active':key,'data-mlsys-focus':''}));}
  if(s.plate){const[x,y,w,h]=PLATE[s.plate];marks.append(svg('rect',{x:x-1.2,y:y-1,width:w+2.4,height:h+2,rx:2,fill:'none',stroke:'#cc5500','stroke-width':1.2,'data-mlsys-focus':''}));}
  for(const m of s.marks){const[x,y,w,h]=PLATE[m];marks.append(svg('rect',{x:x-1.2,y:y-1,width:w+2.4,height:h+2,rx:2,fill:'none',stroke:i===states.length-1?'#5f6b73':'#b11d33','stroke-width':1.2,'stroke-dasharray':'2 1.5','data-mark':String(m),'data-mlsys-focus':''}));}
  for(const[key,[head,value,note,state,noteState]]of Object.entries(s.tags)){
   const[a,b,c]=tagNodes[key];a.textContent=head;b.textContent=value;c.textContent=note;b.setAttribute('fill',state?tone[state]:'#17212b');c.setAttribute('fill',noteState?tone[noteState]:tone['']);c.setAttribute('font-weight',noteState?600:400);[a,b,c].forEach(n=>fit(n,TAG_W-6));
   const card=cards[key];card.children[1].textContent=head;card.children[2].textContent=value;card.children[3].textContent=note;card.dataset.tone=state||'';card.dataset.noteTone=noteState||'';card.classList.toggle('is-active',s.active.includes(key));
  }
  const S=s.S;panelText[0].textContent=s.bf16?'BF16 path':'Loss scale S';panelText[1].textContent=s.bf16?'no scaling':S?(S.exponent?S.S_pow+' = '+S.S_plain:'1'):'not set';panelText[2].textContent=s.status[0];panelText[3].textContent=s.status[1];
  // Phones show the panel beside the drawing without the outer tags; enlarge its text there.
  panelText.forEach((n,k)=>{n.setAttribute('font-size',[6.6,7.4,6.8,6.6][k]*(narrow.matches?1.3:1));fit(n,60);});
  readout.replaceChildren(html('strong','',s.bf16?'BF16 comparison, no loss scale':`S = ${S?S.S_text:'not set'}`),document.createTextNode(` · ${s.status[0]}${s.status[1]?' ('+s.status[1]+')':''} · toy values; IEEE rounding computed in Python.`));
  root.dataset.ledger=JSON.stringify(s.ledger);root.dataset.endpoint=String(i);root.dataset.scale=S?String(S.S):'';root.dataset.update=s.ledger.update||'none';
 }
 const player=window.MLSysOriginalPlayer.create({root,outer,title:'Why scale the loss, and what if S is wrong?',
  scope:`Declared toy values follow one FP32 master weight w = ${D.toy.w32_text}, its gradient g = ${D.toy.g_small_text}, the largest gradient entry ${D.toy.g_large_text}, loss L = ${D.toy.loss_text}, and learning rate η = ${D.toy.eta_text}. All FP16 and BF16 values are real IEEE rounding computed in Python, not in the browser. S = 2¹⁰ and 2¹⁵ are the chapter's examples; 2¹⁴ is one halving. Each tag reports the value stored after its box's step; a real gradient tensor holds a distribution, not two numbers. Framework implementations may vary. Motion is illustrative and does not represent time; no speed or memory claim.`,
  steps,
  // Compact layout: only the drawing sits above the control bar; on phones the center panel
  // (loss scale S and the step's status) stays inside the loop, and the node cards and status
  // readout follow the narration. Intro facts: four scales across the six-stage loop; the
  // tracked g rounds to 0 at S = 1 (trial 0), the largest entry overflows at S = 2¹⁵ (trial 2).
  layout:'compact',intro:`Compare loss scales around the six-stage loop and watch gradient g underflow at S = ${D.trials[0].S_plain} while the largest gradient entry overflows at S = ${D.trials[2].S_pow}.`,
  render(i,{stage,details}){committed=i;if(!mounted){stage.append(source);details.append(grid,readout);mounted=true;}motion.replaceChildren();paint(i);delete root.dataset.pending;},
  transition({to,onFrame}){
   root.dataset.pending=String(to);const token=states[to].token;motion.replaceChildren();if(!token)return;
   const k=token[0],[[x1,y1],[x2,y2]]=ARROW[k],g=svg('g'),label=svg('text',{'text-anchor':'middle','dominant-baseline':'central','font-size':5.6,fill:'#17212b'},token[1]),box=svg('rect',{y:-4.5,height:9,rx:2,fill:'#ffe5cc',stroke:'#8f3b00','stroke-width':.6});
   g.append(box,label);motion.append(g);
   let tw=30;try{tw=Math.max(14,label.getComputedTextLength()+5);}catch(e){}box.setAttribute('x',-tw/2);box.setAttribute('width',tw);
   const L=Math.hypot(x2-x1,y2-y1),dx=(x2-x1)/L,dy=(y2-y1)/L,side=Array.isArray(SIDE[k][0])?SIDE[k][narrow.matches?1:0]:SIDE[k];
   const[px,py,pw,ph]=PLATE[k],corners=[[px,py],[px+pw,py],[px,py+ph],[px+pw,py+ph]];
   const along=corners.map(([x,y])=>(x-x1)*dx+(y-y1)*dy),half=[tw/2,4.5],hAlong=dx?half[0]:half[1],hSide=side[0]?half[0]:half[1];
   const a=Math.min(...along)-hAlong-1,b=Math.max(...along)+hAlong+1,R=5,off=Math.max(...corners.map(([x,y])=>(x-x1)*side[0]+(y-y1)*side[1]))+1+hSide;
   const ease=v=>{v=Math.min(1,Math.max(0,v));return v*v*(3-2*v);};
   onFrame(p=>{const d=p*L,f=ease((d-a+R)/R)*ease((b+R-d)/R);g.setAttribute('transform',`translate(${x1+dx*d+side[0]*off*f} ${y1+dy*d+side[1]*off*f})`);root.dataset.animationProgress=String(p);});
  }});
 return player;
};})();
