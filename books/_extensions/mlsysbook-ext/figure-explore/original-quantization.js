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
   ['Quantize the activations','Divide each activation by its scale and round. The stored integers are 3 and −5; they represent 0.3 and −0.5.'],
   ['Quantize the weights','The weight branch uses its own scale. Its stored integers are 4 and 2, representing 0.4 and 0.2.'],
   ['Feed the integer operands','Both INT8 branches feed the same matrix multiplication. This small example computes one output from two pairs.'],
   ['Accumulate the first product','Multiply 3 by 4. Store the partial sum in the INT32 accumulator, not in an INT8 output.'],
   ['Accumulate the second product','Add −5 times 2 to the existing partial sum. Both contributions have now arrived.'],
   ['Requantize the completed sum','Convert from the accumulator scale to the output scale, round, and clamp to the INT8 range. Here the integer remains 2.'],
   ['Apply the activation','This worked case uses ReLU: negative represented values become zero. The positive value here passes through unchanged.'],
   ['Write the INT8 output','The final stored integer is 2, representing 0.02. Quantization introduced error; this is a numerical example, not a speedup claim.']
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
  let calculation, assumptions, index=0;
  const active=[0,1,3,4,5,5,6,7,8];
  function render(i,{stage,details}) {
   overlay.querySelectorAll('.oq-token').forEach(t=>t.remove());
   index=i;if(!canvas.isConnected)stage.append(canvas,cue);
   if(!calculation){calculation=document.createElement('p');calculation.className='oq-calculation';assumptions=document.createElement('details');const summary=document.createElement('summary');summary.textContent='Example assumptions';const note=document.createElement('p');note.textContent=`Symmetric signed INT8; both zero-points are 0. Input scale ${d.sx}, weight scale ${d.sw}, output scale ${d.sy}. Round to nearest, with ties away from zero; saturate to [−128, 127] after rounding. No bias. Movement illustrates logical data flow, not measured hardware timing. The original activation curves are generic; this example applies ReLU.`;assumptions.append(summary,note);details.append(calculation,assumptions);}
   calculation.textContent=equations[i];
   const pair=p=>`${number(d.qx[p])} × ${number(d.qw[p])}`;
   const values=[vector(d.x),i>=1?vector(d.qx):'—',vector(d.w),i>=2?vector(d.qw):'—',i<3?'—':i===3?`${d.qx.length} pairs`:pair(i===4?0:1),i<4?'empty':String(i===4?d.products[0]:d.accumulator),i>=6?String(d.requantized):'—',i>=7?String(d.output):'—',i>=8?String(d.output):'—'];
   cells.forEach((c,j)=>{c.value.textContent=values[j];c.g.classList.toggle('oq-active',j===active[i]);});
   root.dataset.quantizationStep=String(i);
   if(canvas.scrollWidth>canvas.clientWidth){const c=cells[active[i]];canvas.scrollLeft=Math.max(0,(c.x+c.w/2)/506.511*svg.getBoundingClientRect().width-canvas.clientWidth/2);}
  }
  const routes={1:[[30,13],[102,13]],2:[[30,77],[102,77]],3:[[102,13],[160.6,13],[160.6,45]],4:[[160.6,45],[233,45]],5:[[160.6,45],[233,45]],6:[[233,45],[306,45]],7:[[306,45],[404,45]],8:[[404,45],[477,45]]};
  function transition({from,to,duration,animate,onFrame}) {
   if(to!==from+1||!routes[to])return;
   calculation.textContent=equations[to];
   cells.forEach((c,j)=>c.g.classList.toggle('oq-active',to===3?(j===1||j===3):j===active[to]));
   const tracks=to===3?[{route:routes[to],text:'xq'},{route:[[102,77],[160.6,77],[160.6,45]],text:'wq'}]:[{route:routes[to],text:['','FP x','FP w','',String(d.products[0]),number(d.products[1]),String(d.accumulator),String(d.requantized),String(d.output)][to]}];
   tracks.forEach(({route,text})=>{const token=node('g',{class:'oq-token'});
   token.append(node('rect',{x:-16,y:-7,width:32,height:14,rx:3}));
   token.append(node('text',{'text-anchor':'middle',y:3},text));overlay.append(token);
   animate(token,[{opacity:1},{opacity:1}],{duration,fill:'both'});
   onFrame(progress=>{const f=progress*(route.length-1),p=Math.min(route.length-2,Math.floor(f)),t=f-p,a=route[p],b=route[p+1];token.setAttribute('transform',`translate(${a[0]+(b[0]-a[0])*t} ${a[1]+(b[1]-a[1])*t})`);if(progress>=1)token.remove();});
   });
  }
  return window.MLSysOriginalPlayer.create({root,outer:context.outer,title:'How do floating-point inputs become an INT8 output?',scope:'A two-term dot product, with zero-points 0 and no bias. Requantize, then apply ReLU. The original curves show other activation shapes.',steps,render,transition});
 };
})();
