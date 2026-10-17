/* fig-cnn: explicit single-channel, stride-one, valid cross-correlation.
 * Rows of three MACs advance a deterministic state; interpolation never owns data.
 */
(() => {
  'use strict';
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons.convolution = root => {
    const abort = new AbortController();
    const on = (el, name, fn) => el.addEventListener(name, fn, {signal: abort.signal});
    const make = (tag, cls, text) => { const e = document.createElement(tag); if(cls)e.className=cls; if(text!==undefined)e.textContent=text; return e; };
    const ns='http://www.w3.org/2000/svg';
    const svgEl=(tag,attrs={})=>{const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));return e;};
    const input=Array.from({length:7},(_,r)=>Array.from({length:7},(_,c)=>c<3?1:4));
    const kernel=[[1,0,-1],[1,0,-1],[1,0,-1]];
    const dot=(r,c)=>kernel.reduce((s,row,i)=>s+row.reduce((t,v,j)=>t+v*input[r+i][c+j],0),0);
    const output=Array.from({length:5},(_,r)=>Array.from({length:5},(_,c)=>dot(r,c)));
    if(JSON.stringify(output)!==JSON.stringify(Array.from({length:5},()=>[0,-9,-9,0,0])))throw new Error('Convolution toy arithmetic mismatch');
    let state=0,running=false,timer=null,destroyed=false;
    root.classList.add('mlsys-convolution');
    const intro=make('p','mlsys-conv-intro','Slide one shared 3 × 3 filter across a 7 × 7 input. Watch nine products accumulate into each violet output cell.');
    const scope=make('p','mlsys-conv-scope','Worked example: one input channel, one filter, stride 1, no padding or bias. Integer values reveal a vertical edge. Like common ML convolution operators, this uses cross-correlation (the kernel is not flipped).');
    const controls=make('div','mlsys-conv-controls'),buttons={};
    for(const [key,label] of [['play','Play'],['back','Back'],['next','Next'],['restart','Restart']]){const b=make('button','',label);b.type='button';buttons[key]=b;controls.append(b);}
    const label=make('label','','Sequence '),menu=make('select');menu.id=root.id+'-sequence';label.htmlFor=menu.id;
    const phaseNames=['place window','accumulate row 1','accumulate row 2','accumulate row 3','write output'];
    const opts=[['0','Start: input and filter'],...Array.from({length:10},(_,i)=>[String(i+1),`[0, ${Math.floor(i/5)}] · ${phaseNames[i%5]}`]),...Array.from({length:23},(_,i)=>[String(11+i),`[${Math.floor((i+2)/5)}, ${(i+2)%5}] · nine MACs and write`]),['34','Complete: reuse and cost']];
    opts.forEach(([value,text])=>{const o=make('option','',text);o.value=value;menu.append(o);});label.append(menu);controls.append(label);
    const status=make('p','mlsys-conv-status');status.setAttribute('aria-live','polite');status.setAttribute('aria-atomic','true');
    const diagram=svgEl('svg',{viewBox:'0 0 620 590',role:'img','aria-label':'A violet five by five output above an orange sliding three by three window in the seven by seven input.'});
    const title=(x,y,text)=>{const e=svgEl('text',{x,y,class:'mlsys-conv-svg-title'});e.textContent=text;diagram.append(e);};
    title(230,23,'Output · 5 × 5');title(70,314,'Input · 7 × 7');
    const originIn=[60,425],originOut=[240,103],uIn=[37,-15],vIn=[32,20],uOut=[27,-11],vOut=[25,16];
    const point=(o,u,v,r,c)=>[o[0]+u[0]*c+v[0]*r,o[1]+u[1]*c+v[1]*r];
    const corners=(o,u,v,r,c,n=1)=>[[r,c],[r,c+n],[r+n,c+n],[r+n,c]].map(([a,b])=>point(o,u,v,a,b));
    const pts=p=>p.map(a=>a.join(',')).join(' ');
    const inputCells=[],outputCells=[];
    const addGrid=(n,o,u,v,type,values,collection)=>{for(let r=0;r<n;r++)for(let c=0;c<n;c++){
      const g=svgEl('g'),p=svgEl('polygon',{points:pts(corners(o,u,v,r,c)),class:'mlsys-conv-'+type+'-cell'}),center=point(o,u,v,r+.5,c+.5),t=svgEl('text',{x:center[0],y:center[1]+4,class:'mlsys-conv-value'});t.textContent=type==='input'?values[r][c]:'';g.append(p,t);diagram.append(g);collection.push({p,t,r,c});
    }};
    // Blue projection links stay behind both planes, as in the canonical figure.
    const links=Array.from({length:4},()=>{const e=svgEl('line',{class:'mlsys-conv-link'});diagram.append(e);return e;});
    addGrid(7,originIn,uIn,vIn,'input',input,inputCells);
    const windowPoly=svgEl('polygon',{points:pts(corners(originIn,uIn,vIn,0,0,3)),class:'mlsys-conv-window'});diagram.append(windowPoly);
    addGrid(5,originOut,uOut,vOut,'output',output,outputCells);
    const activeOutput=svgEl('polygon',{points:pts(corners(originOut,uOut,vOut,0,0)),class:'mlsys-conv-output-active'});diagram.append(activeOutput);
    const work=make('div','mlsys-conv-work'),kernelPanel=make('section'),kernelTitle=make('h4','','Shared filter · 9 weights'),kernelGrid=make('div','mlsys-conv-kernel');
    kernel.flat().forEach(v=>kernelGrid.append(make('span','',v)));kernelPanel.append(kernelTitle,kernelGrid,make('p','','The same weights stay in use at all 25 positions.'));
    const mathPanel=make('section'),mathTitle=make('h4','','Current products'),products=make('div','mlsys-conv-products'),productCells=[];
    for(let i=0;i<9;i++){const e=make('span');products.append(e);productCells.push(e);}
    const accumulator=make('p','mlsys-conv-accumulator'),operation=make('p','mlsys-conv-operation');mathPanel.append(mathTitle,products,accumulator,operation);work.append(kernelPanel,mathPanel);
    const narration=make('p','mlsys-conv-narration'),counts=make('p','mlsys-conv-counts');
    const body=make('div','mlsys-conv-body');body.append(diagram,work);
    root.append(intro,scope,controls,status,body,narration,counts,make('p','mlsys-conv-note','MAC and access counts are logical scalar work, not measured memory transactions or timings. Neighboring windows overlap; caching or tiling can capture reuse, but this drawing does not predict speedup. No activation function is applied.'));
    function render(){
      root.classList.toggle('mlsys-conv-paused', !running);
      const complete=state===34,started=state>0,p=complete?24:state>=11?state-9:Math.max(0,Math.floor((state-1)/5)),phase=complete||state>=11?4:state===0?-1:(state-1)%5,r=Math.floor(p/5),c=p%5;
      const rowsDone=Math.max(0,Math.min(3,phase)),sum=kernel.slice(0,rowsDone).reduce((s,row,i)=>s+row.reduce((t,w,j)=>t+w*input[r+i][c+j],0),0);
      inputCells.forEach(e=>{const active=started&&e.r>=r&&e.r<r+3&&e.c>=c&&e.c<c+3;e.p.classList.toggle('mlsys-conv-in-window',active);e.p.classList.toggle('mlsys-conv-current-row',active&&phase>=1&&phase<=3&&e.r===r+phase-1);});
      outputCells.forEach(e=>{const index=e.r*5+e.c,written=complete||index<p||(index===p&&phase===4);e.t.textContent=written?output[e.r][e.c]:'';e.p.classList.toggle('mlsys-conv-written',written);});
      const off=point(originIn,uIn,vIn,r,c),delta=[off[0]-originIn[0],off[1]-originIn[1]];windowPoly.setAttribute('transform',`translate(${delta})`);
      const o=point(originOut,uOut,vOut,r,c);activeOutput.setAttribute('transform',`translate(${o[0]-originOut[0]},${o[1]-originOut[1]})`);
      const wc=corners(originIn,uIn,vIn,r,c,3),oc=corners(originOut,uOut,vOut,r,c);links.forEach((e,i)=>{e.setAttribute('x1',wc[i][0]);e.setAttribute('y1',wc[i][1]);e.setAttribute('x2',oc[i][0]);e.setAttribute('y2',oc[i][1]);});
      windowPoly.style.opacity=activeOutput.style.opacity=started&&!complete?'1':'0';links.forEach(e=>e.style.opacity=started&&!complete?'.6':'0');
      productCells.forEach((e,i)=>{const a=Math.floor(i/3),b=i%3,v=input[r+a][c+b],w=kernel[a][b];e.textContent=`${v} × ${w} = ${v*w}`;e.classList.toggle('mlsys-conv-product-done',a<rowsDone);e.classList.toggle('mlsys-conv-product-current',phase>=1&&phase<=3&&a===phase-1);});
      const macs=complete?225:p*9+rowsDone*3,writes=complete?25:p+(phase===4?1:0);
      status.textContent=complete?'Complete · all 25 outputs written':!started?'Ready · 7 − 3 + 1 = 5 positions on each axis':`Output [${r}, ${c}] · ${(state>=11?'nine MACs and write':phaseNames[phase])}`;
      accumulator.textContent=`Accumulator: ${started?sum:0}${phase===4?' → output ['+r+', '+c+']':''}`;
      const terms=phase>=1&&phase<=3?kernel[phase-1].map((w,j)=>`${input[r+phase-1][c+j]} × (${w})`).join(' + '):'';
      operation.textContent=terms?`Add this row: ${terms}.` : complete?'Each output sums nine products.':phase===4?(state>=11?'Repeat the same nine-product sum, then write once. All nine current products are shown.':'Write once after all nine products; reset the accumulator for the next window.'):'Start this output with an accumulator of zero.';
      narration.textContent=complete?'The vertical intensity change produces −9 in two output columns; uniform windows produce 0. The filter is shared, so 25 outputs require only 9 distinct weights. There are 225 logical input-value uses across 49 input elements: overlapping windows create reuse opportunities.':!started?'The input contains 1 in columns 0–2 and 4 in columns 3–6. The filter subtracts the right column from the left at each of three rows; signed values expose the direction of the edge.':phase===0?(c>0?'Slide one input column to the right. Six of the nine input values overlap the preceding window; the filter does not change.':r>0?'Move to the next output row and reset the horizontal position. The same filter starts a new window; no padding is introduced.':'The first orange window covers input rows 0–2 and columns 0–2. Its blue links identify output [0, 0].'):phase===4?`${state>=11?'The remaining positions repeat the same nine-MAC operation in one step. ':''}All three filter rows have contributed. Store ${output[r][c]} at output [${r}, ${c}]; this completed output no longer needs its local partial sum.`:'The highlighted filter row contributes three multiply-accumulates. Retain the running sum until the remaining rows have contributed; partial sums are not completed outputs.';
      counts.textContent=`Completed: ${macs} / 225 MACs · ${writes} / 25 output writes. Single output: 3 × 3 × 1 input channel = 9 MACs.`;
      menu.value=String(state);buttons.back.disabled=state===0;buttons.next.disabled=complete;buttons.play.textContent=running?'Pause':complete?'Replay':'Play';
      root.dataset.convolutionState=state;root.dataset.convolutionMacs=macs;root.dataset.convolutionWrites=writes;
    }
    function pause(){running=false;clearTimeout(timer);timer=null;if(!destroyed)render();}
    function schedule(){timer=setTimeout(()=>{if(!running||destroyed)return;state++;if(state===34)running=false;render();if(running)schedule();},1600);}
    on(buttons.play,'click',()=>{if(running)return pause();if(state===34)state=0;running=true;render();schedule();});
    on(buttons.back,'click',()=>{pause();state=Math.max(0,state-1);render();});on(buttons.next,'click',()=>{pause();state=Math.min(34,state+1);render();});on(buttons.restart,'click',()=>{pause();state=0;render();});
    on(menu,'change',()=>{const next=Number(menu.value);pause();state=next;render();});on(document,'visibilitychange',()=>{if(document.hidden)pause();});
    const observer=typeof IntersectionObserver==='undefined'?null:new IntersectionObserver(entries=>{if(!entries[0].isIntersecting&&running)pause();},{threshold:.1});observer?.observe(root);
    render();return{pause,destroy(){if(destroyed)return;pause();destroyed=true;observer?.disconnect();abort.abort();root.replaceChildren();}};
  };
})();
