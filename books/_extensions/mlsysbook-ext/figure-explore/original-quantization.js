/* The chapter's verified TikZ artwork, with a declared numerical example. */
(() => {
 'use strict';
 window.MLSysFigureLessons = window.MLSysFigureLessons || {};
 window.MLSysFigureLessons['original-quantization'] = async (root, context) => {
  const original = context.verifiedArtwork || context.artwork;
  if (!original || !window.MLSysOriginalPlayer || !window.MLSysOriginalQuantizationData) throw Error('Original quantization dependencies unavailable');
  const d=window.MLSysOriginalQuantizationData, ns='http://www.w3.org/2000/svg';
  const number=value=>String(value).replace(/-/g,'−'), vector=values=>'['+values.map(number).join(', ')+']';
  root.classList.add('mlsys-original-quantization');
  const svg=original.cloneNode(true);svg.setAttribute('role','img');svg.setAttribute('aria-label','Original quantized matrix multiplication pipeline with synchronized example values');
  // Reserve a clear payload-label lane; the canonical drawing stays at its original coordinates.
  const {x:artX,y:artY,width:artWidth,height:artHeight}=svg.viewBox.baseVal;svg.setAttribute('viewBox',`${artX} ${artY-24} ${artWidth} ${artHeight+24}`);svg.setAttribute('width',String(artWidth));svg.setAttribute('height',String(artHeight+24));
  const overlay=document.createElementNS(ns,'g');overlay.classList.add('oq-overlay');svg.append(overlay);
  const canvas=document.createElement('div');canvas.className='oq-canvas';canvas.tabIndex=0;canvas.setAttribute('aria-label','Pipeline diagram. Scroll sideways to inspect on a narrow screen.');canvas.append(svg);
  const cue=document.createElement('p');cue.className='oq-scroll-cue';cue.textContent='↔ Swipe the diagram to inspect. Each step centers the current operation.';
  const boxes=[
   [0.68,0.37,57.72,25.46,'Activation input','#fbdafb'],[73.28,0.37,57.72,25.46,'Quantization','#fbdafb'],
   [0.68,64.09,57.72,25.46,'Weight input','#fbdafb'],[73.28,64.09,57.72,25.46,'Quantization','#fbdafb'],
   [131.75,32.23,57.72,25.46,'Integer matmul','#d1f3ff'],[204.36,32.23,57.72,25.46,'INT32 sum','#fde2f0'],
   [276.96,32.23,57.72,25.46,'Requantization','#dbfda6'],[375.03,32.23,57.72,25.46,'Activation: ReLU','#e9deda'],
   [447.63,32.23,57.72,25.46,'INT8 output','#dbfda6']
  ];
  const node=(tag,attrs={},text)=>{const n=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;return n;};
  const cells=boxes.map(([x,y,w,h,label,fill])=>{
   const g=node('g',{'data-oq-box':label});g.append(node('rect',{x:x+0.9,y:y+0.9,width:w-1.8,height:h-1.8,fill}));
   g.append(node('text',{x:x+w/2,y:y+8,'text-anchor':'middle',class:'oq-label'},label));
   const value=node('text',{x:x+w/2,y:y+20,'text-anchor':'middle',class:'oq-value'},'');g.append(value);overlay.append(g);return {g,value,x,y,w,h};
  });
  // The original generic activation curves remain visible and explicitly distinguished.
  const steps=[
   ['Start with two floating-point inputs','Follow one dot product through the original pipeline. The top branch carries activations; the lower branch carries weights.'],
   ['Quantize the activations','Divide each activation by its scale and round. The converted integers appear when this action finishes.'],
   ['Quantize the weights','The weight branch uses its own scale. Follow its inputs into the quantizer; the converted integers appear on arrival.'],
   ['Feed the integer operands','Both INT8 branches feed the same matrix multiplication. This small example computes one output from two pairs.'],
   ['Accumulate the first product','Send the first product to the INT32 accumulator. The stored sum changes only when the product arrives.'],
   ['Accumulate the second product','The first partial sum remains stored until the second product arrives. Addition completes the reduction before conversion to INT8.'],
   ['Requantize the completed sum','All products must arrive before this conversion. Rescale the completed sum, round, and clamp to the INT8 range.'],
   ['Apply the activation','This worked case uses ReLU: negative represented values become zero. The positive value here passes through unchanged.'],
   ['Write the INT8 output','Write the completed integer output. Its real value depends on the output scale. This numerical example does not establish a speedup.']
  ].map(([label,description])=>({label,description}));
  const equations=[
   `x = ${vector(d.x)}; w = ${vector(d.w)}`,
   `round(x / ${d.sx}) = ${vector(d.qx)}`,
   `round(w / ${d.sw}) = ${vector(d.qw)}`,
   'INT8 operands → integer multiply → INT32 accumulation',
   `0 + (${d.qx[0]} × ${d.qw[0]}) = ${d.products[0]}`,
   `${d.products[0]} + (${number(d.qx[1])} × ${d.qw[1]}) = ${d.accumulator}`,
   `round(${d.accumulator} × ${d.sx} × ${d.sw} / ${d.sy}) = ${d.requantized}`,
   `max(0, ${d.requantized}) = ${d.output}`,
   `${d.output} × ${d.sy} = ${d.reconstructed}; floating-point reference = ${d.float_reference}`
  ];
  let calculation, calculationLabel, calculationValue, assumptions;
  const active=[0,1,3,4,5,5,6,7,8];
  function centerBoxes(indices) {
   if(canvas.scrollWidth<=canvas.clientWidth)return;
   const left=Math.min(...indices.map(j=>cells[j].x)),right=Math.max(...indices.map(j=>cells[j].x+cells[j].w));
   canvas.scrollLeft=Math.max(0,((left+right)/2-artX)/artWidth*svg.getBoundingClientRect().width-canvas.clientWidth/2);
  }
  function render(i,{stage,details}) {
   overlay.querySelectorAll('.oq-token').forEach(t=>t.remove());
   if(!calculation){calculation=document.createElement('p');calculation.className='oq-calculation';calculationLabel=document.createElement('span');calculationLabel.className='oq-calculation-label';calculationValue=document.createElement('span');calculation.append(calculationLabel,calculationValue);assumptions=document.createElement('details');const summary=document.createElement('summary');summary.textContent='Example assumptions';const note=document.createElement('p');note.textContent=`Symmetric signed INT8; both zero-points are 0. Input scale ${d.sx}, weight scale ${d.sw}, output scale ${d.sy}. Round to nearest, with ties away from zero; saturate to [−128, 127] after rounding. No bias. Movement illustrates logical data flow, not measured hardware timing. The original activation curves are generic; this example applies ReLU.`;assumptions.append(summary,note);details.append(assumptions);}
   if(!canvas.isConnected)stage.append(calculation,canvas,cue);
   calculationLabel.textContent=i===0?'Initial values':i===3?'Operands ready · INT32 sum initialized to zero':'Result now stored';
   calculationValue.textContent=equations[i];
   delete root.dataset.oqPending;
   const pair=p=>`${number(d.qx[p])} × ${number(d.qw[p])}`;
   const values=[vector(d.x),i>=1?vector(d.qx):'—',vector(d.w),i>=2?vector(d.qw):'—',i<3?'—':i===3?`${d.qx.length} pairs`:pair(i===4?0:1),i<3?'not started':i===3?'0':String(i===4?d.products[0]:d.accumulator),i>=6?String(d.requantized):'—',i>=7?String(d.output):'—',i>=8?String(d.output):'—'];
   cells.forEach((c,j)=>{c.value.textContent=values[j];c.g.classList.toggle('oq-active',j===active[i]);c.g.classList.remove('oq-pending','oq-source');});
   root.dataset.quantizationStep=String(i);
   centerBoxes([active[i]]);
  }
  // Dock outside the box borders. The packet follows the original arrows;
   // its payload badge stays in an empty lane, away from every label.
  const rightPort=i=>[cells[i].x+cells[i].w+3.5,cells[i].y+cells[i].h/2];
  const leftPort=i=>[cells[i].x-3.5,cells[i].y+cells[i].h/2];
  const routes={1:[rightPort(0),leftPort(1)],2:[rightPort(2),leftPort(3)],3:[rightPort(1),[160.6,13.1],[160.6,28.73]],4:[rightPort(4),leftPort(5)],5:[rightPort(4),leftPort(5)],6:[rightPort(5),leftPort(6)],7:[rightPort(6),leftPort(7)],8:[rightPort(7),leftPort(8)]};
  function transition({from,to,duration,animate,onFrame}) {
   if(to!==from+1||!routes[to])return;
   const source=to===3?[1,3]:[({1:0,2:2,4:4,5:4,6:5,7:6,8:7})[to]];
   root.dataset.oqPending=String(to);
   calculationLabel.textContent='In transit · dashed box is waiting';
   const waiting=[
    '', `Activation inputs → quantizer; converted values pending`,
    `Weight inputs → quantizer; converted values pending`,
    `INT8 operands → integer matmul; INT32 accumulation not started`,
    `INT32 sum held at 0; incoming product ${d.qx[0]} × ${d.qw[0]} = ${d.products[0]}`,
    `INT32 sum held at ${d.products[0]}; incoming product ${number(d.qx[1])} × ${d.qw[1]} = ${number(d.products[1])}`,
    `Completed INT32 sum ${d.accumulator} → requantizer; INT8 result pending`,
    `Requantized integer ${d.requantized} → ReLU; activation result pending`,
    `Activated integer ${d.output} → output; write pending`
   ];
   calculationValue.textContent=waiting[to];
   // Operand selection is read-only; only endpoint render commits the sum.
   if(to===4||to===5){const pairIndex=to-4;cells[4].value.textContent=`${number(d.qx[pairIndex])} × ${number(d.qw[pairIndex])}`;}
   cells.forEach((c,j)=>{c.g.classList.remove('oq-active');c.g.classList.toggle('oq-source',source.includes(j));c.g.classList.toggle('oq-pending',j===active[to]);});
   centerBoxes([...source,active[to]]);
   const tracks=to===3?[{route:routes[to],text:'xq',lane:-12},{route:[rightPort(3),[160.6,76.82],[160.6,61.19]],text:'wq',lane:103}]:[{route:routes[to],lane:to===2?103:to===1?-12:13,text:['','FP x','FP w','',String(d.products[0]),number(d.products[1]),String(d.accumulator),String(d.requantized),String(d.output)][to]}];
   tracks.forEach(({route,text,lane})=>{const token=node('g',{class:'oq-token'});
   const packet=node('rect',{class:'oq-packet',x:-2.5,y:-2.5,width:5,height:5,rx:1});
   const badge=node('g',{class:'oq-token-badge'});badge.append(node('rect',{x:-16,y:-7,width:32,height:14,rx:3}));
   const pointer=node('path',{class:'oq-token-pointer'});badge.append(pointer,node('text',{'text-anchor':'middle',y:3},text));token.append(packet,badge);overlay.append(token);
   animate(token,[{opacity:1},{opacity:1}],{duration,fill:'both'});
   onFrame(progress=>{const f=progress*(route.length-1),p=Math.min(route.length-2,Math.floor(f)),t=f-p,a=route[p],b=route[p+1],x=a[0]+(b[0]-a[0])*t,y=a[1]+(b[1]-a[1])*t;token.setAttribute('transform',`translate(${x} ${y})`);badge.setAttribute('transform',`translate(0 ${lane-y})`);pointer.setAttribute('d',lane<y?'M -2 7 L 0 9 L 2 7 Z':'M -2 -7 L 0 -9 L 2 -7 Z');if(progress>=1)token.remove();});
   });
  }
  return window.MLSysOriginalPlayer.create({root,outer:context.outer,title:'How do floating-point inputs become an INT8 output?',scope:'A two-term dot product, with zero-points 0 and no bias. Requantize, then apply ReLU. The original curves show other activation shapes.',steps,render,transition});
 };
})();
