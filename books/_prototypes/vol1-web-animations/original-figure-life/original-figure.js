/* One review prototype: move the original TikZ paths, not a replacement diagram. */
(() => {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const node = (tag, text, cls) => {
    const el = document.createElement(tag);
    if (text) el.textContent = text;
    if (cls) el.className = cls;
    return el;
  };
  function start(outer) {
    if (!window.MLSysFigureViews) return;
    const figure = outer.querySelector('figure'), caption = figure.querySelector('figcaption');
    const staticNodes = [...figure.childNodes].filter(n => n !== caption);
    const diagram = node('div'), panel = node('section', '', 'life-panel');
    panel.tabIndex = 0;
    panel.setAttribute('aria-label', 'Convolution on the original drawing');
    const svg = outer.querySelector('template[data-original-art]').content.querySelector('svg').cloneNode(true);
    svg.removeAttribute('width'); svg.removeAttribute('height');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'The original input grid, receptive field, blue connections, and output grid, animated in place.');
    // Keep clip paths local to this copy of the artwork.
    svg.querySelectorAll('[id]').forEach(el => {
      const old = el.id, replacement = 'life-' + old;
      svg.querySelectorAll('[clip-path]').forEach(n => {
        if (n.getAttribute('clip-path') === `url(#${old})`) n.setAttribute('clip-path', `url(#${replacement})`);
      });
      el.id = replacement;
    });
    const paths = [...svg.querySelectorAll('path')];
    const input = paths.filter(p => p.hasAttribute('stroke-dasharray'));
    const orange = paths.filter(p => p.getAttribute('fill') === 'rgb(100%, 50%, 0%)');
    const violet = paths.filter(p => p.getAttribute('fill') === 'rgb(89.99939%, 79.998779%, 89.99939%)');
    const links = paths.filter(p => p.getAttribute('stroke') === 'rgb(0%, 0%, 100%)');
    if (input.length !== 49 || orange.length !== 9 || violet.length !== 50 || links.length !== 4)
      throw new Error('Original artwork changed: review the geometric mapping.');
    const polygon = p => {
      const values = p.getAttribute('d').match(/-?\d*\.?\d+/g).map(Number);
      const matrix = new DOMMatrix(p.getAttribute('transform') || undefined);
      return [0, 2, 4, 6].map(i => new DOMPoint(values[i], values[i + 1]).matrixTransform(matrix));
    };
    const add = (a,b) => ({x:a.x+b.x, y:a.y+b.y});
    const minus = (a,b) => ({x:a.x-b.x, y:a.y-b.y});
    const scale = (a,k) => ({x:a.x*k, y:a.y*k});
    const center = points => scale(points.reduce(add, {x:0,y:0}), 1 / points.length);
    const grid = cell => {
      const points = polygon(cell);
      return {origin:points[0], x:minus(points[1],points[0]), y:minus(points[3],points[0])};
    };
    const lower = grid(input[0]), upper = grid(violet[0]), orangeOrigin=polygon(orange[0])[0];
    // Uniform enlargement preserves parallel cell edges in both planes.
    const outputPoints=violet.flatMap(p=>polygon(p));
    const outputScale=1.25, outputBottom=Math.max(...outputPoints.map(p=>p.y));
    const outputCenter=(Math.min(...outputPoints.map(p=>p.x))+Math.max(...outputPoints.map(p=>p.x)))/2;
    const project=p=>({x:outputCenter+(p.x-outputCenter)*outputScale,y:outputBottom+(p.y-outputBottom)*outputScale});
    upper.origin=project(upper.origin); upper.x=scale(upper.x,outputScale); upper.y=scale(upper.y,outputScale);
    const top=Math.min(...violet.flatMap(p=>polygon(p).map(v=>project(v).y)))-3;
    const box=svg.viewBox.baseVal;
    svg.setAttribute('viewBox',`${box.x} ${top} ${box.width} ${box.height-top}`);
    const outputPlane=document.createElementNS(NS,'g');
    outputPlane.setAttribute('transform',`translate(${outputCenter*(1-outputScale)} ${outputBottom*(1-outputScale)}) scale(${outputScale})`);
    const highlighted=paths.filter(p=>['rgb(75%, 50%, 75%)','rgb(64.99939%, 29.998779%, 64.99939%)'].includes(p.getAttribute('fill')));
    [...svg.children].filter(el=>violet.includes(el)||highlighted.includes(el)||/clip-(?:0|1|4|5)\)/.test(el.getAttribute('clip-path')||'')).forEach(el=>outputPlane.append(el));

    const originalOutput=polygon(paths.find(p=>p.getAttribute('fill')==='rgb(64.99939%, 29.998779%, 64.99939%)')).map(project);
    const at = (g,x,y) => add(g.origin, add(scale(g.x,x), scale(g.y,y)));
    const corners = (g,x,y,size=1) => [at(g,x,y),at(g,x+size,y),at(g,x+size,y+size),at(g,x,y+size)];
    const points = a => a.map(p => `${p.x},${p.y}`).join(' ');
    // Hide the two static highlighted output cells; the original base grid stays.
    paths.filter(p => ['rgb(75%, 50%, 75%)','rgb(64.99939%, 29.998779%, 64.99939%)'].includes(p.getAttribute('fill'))).forEach(p => p.style.visibility='hidden');
    const make = (tag,attrs) => {
      const el = document.createElementNS(NS,tag);
      Object.entries(attrs).forEach(([key,value]) => el.setAttribute(key,value));
      return el;
    };
    const windowGroup = make('g',{});
    const firstOrange = orange[0];
    firstOrange.before(windowGroup);
    orange.forEach(p => windowGroup.append(p));
    const outputLayer = make('g',{}), particles = make('g',{'aria-hidden':'true'});
    const valueLayer=make('g',{'data-values-on-diagram':''});
    svg.append(outputPlane,outputLayer,valueLayer,particles);
    const controls = node('div','','life-controls'), buttons={};
    for (const [id,label] of [['next','Next'],['back','Back'],['play','Play'],['restart','Restart']]) {
      const b=node('button',label); b.type='button'; b.dataset.control=id;
      b.setAttribute('aria-keyshortcuts',({next:'N',back:'P',play:'K',restart:'R'})[id]);
      controls.append(b); buttons[id]=b;
    }
    const label=node('label','Step ','life-sequence'), menu=node('select');
    menu.setAttribute('aria-label','Animation sequence'); label.append(menu);
    const narration=node('div','','life-narration'), heading=node('strong'), explanation=node('p');
    narration.setAttribute('aria-live','polite'); narration.append(heading,explanation);
    const canvas=node('div','','life-art'); canvas.append(svg);
    const workspace=node('div','','life-workspace'), inset=node('details','','life-values');
    inset.setAttribute('aria-label','Detailed calculation for the selected window');
    const example=JSON.parse(outer.querySelector('[data-example-values]').textContent);
    const insetTitle=node('summary','Inspect the calculation');
    const sourceLink=node('a','Illustrative values from the vertical-line example below','life-value-source'); sourceLink.href='#'+example.source_anchor;
    const valueScope=node('p',example.scope,'life-value-scope'), valuesBody=node('div');
    inset.append(insetTitle,valueScope,valuesBody);
    const calculation=node('section','','life-calculation');
    calculation.setAttribute('aria-label','Calculation linked to the highlighted row');
    const calculationLabel=node('strong','Multiply matching cells, then add'), rowControls=node('div','','life-row-controls');
    rowControls.setAttribute('role','group'); rowControls.setAttribute('aria-label','Inspect a row of the current window');
    const rowButtons=[];let selectedRow=0;
    for(let row=0;row<example.weights.length;row++) {
      const button=node('button',`Row ${row+1}`);button.type='button';rowControls.append(button);rowButtons.push(button);
      button.addEventListener('click',()=>{selectedRow=row;paintCalculation(states[index]);paintDiagramValues(states[index]);});
    }
    const equation=node('p','','life-row-equation'), total=node('p','','life-calculation-total');
    calculation.append(calculationLabel,rowControls,equation,total);
    workspace.append(canvas,calculation,sourceLink,inset);
    const keys=node('p','Keyboard: N Next · P Back · K Play/Pause · R Restart. Focus the animation to use shortcuts.','life-keys');
    const legend=node('div','','life-legend');
    for (const [text,color] of [['Input window','#ff8000'],['Input contributions','#2455d6'],['Output activations','#a64ca6']]) {
      const chip=node('span',text); chip.style.setProperty('--chip',color); legend.append(chip);
    }
    const progress=node('p','','life-progress');
    panel.append(controls,keys,label,narration,workspace,legend,progress);
    const movement=Number(outer.dataset.movementMs), hold=Number(outer.dataset.readingPauseMs);
    if (!(movement>0) || !(hold>=0)) throw new Error('Missing build-time timing.');
    const media=matchMedia('(prefers-reduced-motion: reduce)');
    const states=[{pose:true,title:'The drawing in the chapter',text:'The orange window in the lower input grid determines the violet cell above it. The blue lines connect that window to its output.'}];
    for (let row=0; row<5; row++) for (let col=0; col<5; col++) {
      states.push({row,col,phase:'move',title:`Position (${row}, ${col}): move the window`,text:'The same 3 × 3 filter moves to this input window. Only the window position changes; the filter weights are reused.'});
      states.push({row,col,phase:'combine',title:`Position (${row}, ${col}): combine the inputs`,text:'Each of the nine input values is multiplied by its filter weight. The contributions are summed to form one output activation.'});
      states.push({row,col,phase:'write',title:`Position (${row}, ${col}): store the output`,text:'One activation is now stored in the corresponding output cell. The other input values remain available for overlapping windows.'});
    }
    states.forEach((s,i) => {const o=node('option',`${i+1}. ${s.title}`);o.value=i;menu.append(o);});
    let index=0, completed=0, running=false, moving=false, timer=null, effects=[], finishAction=null, motionFrame=null, motionTick=null;
    const coordinates=s => {
      if(s.pose) return {window:[orangeOrigin,add(orangeOrigin,scale(lower.x,3)),add(orangeOrigin,scale(add(lower.x,lower.y),3)),add(orangeOrigin,scale(lower.y,3))],output:originalOutput,shift:{x:0,y:0}};
      const {row,col}=s, window=corners(lower,col,row,3);
      return {window,output:corners(upper,col,row),shift:minus(window[0],orangeOrigin)};
    };
    function applyPose(s) { applyCoordinates(coordinates(s)); }
    function applyCoordinates(c) {
      windowGroup.setAttribute('transform',`translate(${c.shift.x} ${c.shift.y})`);
      links.forEach((link,i) => {
        link.removeAttribute('transform');
        link.setAttribute('d',`M ${c.window[i].x} ${c.window[i].y} L ${c.output[i].x} ${c.output[i].y}`);
      });
    }
    function drawOutputs(s) {
      outputLayer.replaceChildren();
      const count=s.pose ? 0 : s.row*5+s.col+(s.phase==='write'&&!moving?1:0);
      for(let n=0;n<count;n++) outputLayer.append(make('polygon',{points:points(corners(upper,n%5,Math.floor(n/5))),fill:'#a64ca6',stroke:'#000','stroke-width':'.24'}));
      if(s.pose) outputLayer.append(make('polygon',{points:points(originalOutput),fill:'#a64ca6',stroke:'#000','stroke-width':'.24'}));
      progress.textContent=s.pose ? 'One input window → one output activation. The added numbers use the section’s vertical-line example.' : `${count} of 25 output activations stored · overlapping windows reuse input values.`;
      panel.dataset.outputs=String(count);
    }
    function paint() {
      const s=states[index]; if(!moving){applyPose(s);drawOutputs(s);}
      paintValues(s);
      paintCalculation(s);
      paintDiagramValues(s);
      heading.textContent=s.title; explanation.textContent=s.text; menu.value=String(index);
      buttons.back.disabled=index===0; buttons.restart.disabled=index===0&&!moving;
      buttons.next.disabled=index===states.length-1&&!moving;
      buttons.next.textContent=moving?'Finish':'Next'; buttons.play.textContent=running?'Pause':index===states.length-1?'Replay':'Play';
      Object.assign(panel.dataset,{step:String(index),committed:String(completed),moving:String(moving),running:String(running)});
    }
    function paintDiagramValues(s) {
      valueLayer.replaceChildren();
      const text=(content,x,y,size,fill,attrs={})=>{
        const t=make('text',{x,y,'font-size':size,fill,'text-anchor':'middle','dominant-baseline':'central','font-family':'system-ui, sans-serif',...attrs});
        t.textContent=String(content);valueLayer.append(t);return t;
      };
      for(let row=0;row<example.inputs.length;row++)for(let col=0;col<example.inputs[row].length;col++) {
        const c=center(corners(lower,col,row));
        const active=!s.pose&&!(moving&&s.phase==='move')&&row>=s.row&&row<s.row+3&&col>=s.col&&col<s.col+3;
        text(example.inputs[row][col],c.x,c.y,4.4,active?'#161616':'#475569',{'data-input-row':row,'data-input-col':col,'font-weight':active?'700':'400','paint-order':'stroke',stroke:active?'#ff8000':'#fff','stroke-width':'.35'});
      }
      const count=Number(panel.dataset.outputs||0);
      for(let n=0;n<count;n++) {
        const value=example.positions[n],c=center(corners(upper,value.column,value.row));
        text(value.output,c.x,c.y,value.output<0?3.7:4.1,'#fff',{'font-weight':'650','data-output-index':n,'paint-order':'stroke',stroke:'#6e246e','stroke-width':'.12'});
      }
      if(!s.pose&&!(moving&&s.phase==='move')) {
        valueLayer.append(make('polygon',{points:points([at(lower,s.col,s.row+selectedRow),at(lower,s.col+3,s.row+selectedRow),at(lower,s.col+3,s.row+selectedRow+1),at(lower,s.col,s.row+selectedRow+1)]),fill:'none',stroke:'#185c9e','stroke-width':'.65','data-focused-row':selectedRow}));
      }
      const kernelX=96, kernelY=top+10, cell=8;
      text('Shared filter',kernelX+cell*1.5,kernelY-5,3.6,'#222',{'font-weight':'600'});
      example.weights.forEach((row,r)=>row.forEach((weight,c)=>{
        valueLayer.append(make('rect',{x:kernelX+c*cell,y:kernelY+r*cell,width:cell,height:cell,fill:(!s.pose&&r===selectedRow)?'#d9ecff':'#f4f5f7',stroke:'#7c8490','stroke-width':'.18'}));
        text(weight,kernelX+(c+.5)*cell,kernelY+(r+.5)*cell,4.6,'#222',{'data-kernel-row':r,'data-kernel-col':c});
      }));
      const ready=!s.pose&&(s.phase==='write'||(s.phase==='combine'&&!moving));
      if(ready) {
        const value=example.positions.find(p=>p.row===s.row&&p.column===s.col);
        const x=94,y=kernelY+cell*3+5;
        valueLayer.append(make('rect',{x,y,width:30,height:10,rx:1.5,fill:'#eef0ff',stroke:'#7662a2','stroke-width':'.25'}));
        text(`Sum = ${value.output}`,x+15,y+5,4.2,'#46236d',{'data-diagram-sum':value.output,'font-weight':'600'});
      }
    }
    function paintCalculation(s) {
      const waiting=s.pose||(moving&&s.phase==='move');
      calculationLabel.textContent=waiting?'Multiply matching cells, then add':`Row ${selectedRow+1}: multiply matching cells`;
      rowButtons.forEach((button,row)=>{button.disabled=waiting;button.setAttribute('aria-pressed',String(selectedRow===row));});
      equation.replaceChildren();
      if(waiting) {
        equation.textContent=s.pose?'Next selects the first window. Then each input is paired with the weight in the same position.':'The window is moving. Its calculation appears when it arrives.';
        total.textContent='';return;
      }
      const value=example.positions.find(p=>p.row===s.row&&p.column===s.col);
      const ready=s.phase==='write'||(s.phase==='combine'&&!moving);
      value.patch[selectedRow].forEach((input,col)=>{
        if(col)equation.append(node('span',' + ','life-math-sign'));
        const term=node('span','','life-product-term');
        term.append(node('span',String(input),'life-input-number'),node('span',' × '));
        const weight=example.weights[selectedRow][col];
        term.append(node('span',weight<0?`(${weight})`:String(weight),'life-weight-number'));
        equation.append(term);
      });
      equation.append(node('strong',ready?` = ${value.row_sums[selectedRow]}`:' = ?'));
      total.textContent=ready?`Add the three row totals: ${value.row_sums.map(x=>x<0?`(${x})`:x).join(' + ')} = ${value.output}.`:'The blue outline and blue filter row mark the matching cells. Combine evaluates all three rows.';
      calculation.dataset.row=String(selectedRow);
      calculation.dataset.ready=String(ready);
    }
    function paintValues(s) {
      valuesBody.replaceChildren();
      delete inset.dataset.output;
      if(s.pose || (moving&&s.phase==='move')) {
        valuesBody.append(node('p',s.pose?'Advance to select an input window.':'Moving the window. Its values appear when it arrives.','life-value-status'));
        return;
      }
      const value=example.positions.find(p=>p.row===s.row&&p.column===s.col);
      if(!value)throw new Error('Missing worked window.');
      const matrix=(title,rows,cls)=>{
        const box=node('div','',cls),heading=node('p',title),table=node('table');
        table.setAttribute('aria-label',title);
        rows.forEach(row=>{const tr=node('tr');row.forEach(x=>tr.append(node('td',String(x))));table.append(tr);});
        box.append(heading,table);return box;
      };
      valuesBody.append(matrix(`Orange window (${s.row}, ${s.col})`,value.patch,'life-patch'));
      valuesBody.append(matrix('Shared filter — unchanged',example.weights,'life-filter'));
      const ready=s.phase==='write'||(s.phase==='combine'&&!moving);
      if(!ready) {
        valuesBody.append(node('p',moving?'Combining the weighted inputs…':'Ready to multiply and sum.','life-value-status'));
        return;
      }
      const terms=node('div','','life-products');
      value.patch.forEach((row,r)=>{
        const products=row.map((x,c)=>`${x} × ${example.weights[r][c]<0?'('+example.weights[r][c]+')':example.weights[r][c]}`);
        terms.append(node('p',products.join(' + ')+` = ${value.row_sums[r]}`));
      });
      const rowTotals=value.row_sums.map(x=>x<0?'('+x+')':String(x));
      terms.append(node('p',rowTotals.join(' + ')+` = ${value.output}`));
      valuesBody.append(terms);
      const stored=s.phase==='write'&&!moving;
      const result=node('p',stored?`Output[${s.row}, ${s.col}] = ${value.output}`:`Sum ready: ${value.output}`,'life-value-result');
      result.dataset.value=String(value.output);
      valuesBody.append(result);
      inset.dataset.output=String(value.output);
    }
    function cancel() {
      clearTimeout(timer); timer=null;
      if(motionFrame!==null)cancelAnimationFrame(motionFrame);motionFrame=null;motionTick=null;
      effects.forEach(e=>e.cancel()); effects=[]; particles.replaceChildren(); finishAction=null;
    }
    function pause() {
      clearTimeout(timer);timer=null;running=false;effects.forEach(e=>e.pause());
      if(motionFrame!==null)cancelAnimationFrame(motionFrame);motionFrame=null;paint();
    }
    function seek(i) {cancel();running=false;moving=false;index=completed=i;paint();}
    function schedule() {if(running&&index<states.length-1) timer=setTimeout(()=>advance(true),hold);}
    function advance(autoplay=false) {
      if(moving){running=false;finishAction?.();return;}
      if(index===states.length-1)return;
      clearTimeout(timer);timer=null;
      const before=states[index];index++;running=autoplay;moving=true;
      const s=states[index];applyPose(s);drawOutputs(s);paint();
      const duration=media.matches?0:movement;
      const finish=()=>{
        cancel();moving=false;completed=index;if(index===states.length-1)running=false;paint();schedule();
      };
      finishAction=finish;
      if(!duration){finish();return;}
      const animate=(el,frames)=>{
        const animation=el.animate(frames,{duration,easing:'ease-in-out',fill:'forwards'});effects.push(animation);return animation;
      };
      if(s.phase==='move') {
        applyCoordinates(coordinates(before));
      } else if(s.phase==='combine') {
        const target=center(coordinates(s).output);
        for(let row=0;row<3;row++)for(let col=0;col<3;col++) {
          const source=center(corners(lower,s.col+col,s.row+row));
          const dot=make('circle',{cx:0,cy:0,r:'.72',fill:'#2455d6',stroke:'#fff','stroke-width':'.15'});particles.append(dot);
          animate(dot,[{transform:`translate(${source.x}px,${source.y}px)`,opacity:1},{offset:.8,opacity:1},{transform:`translate(${target.x}px,${target.y}px)`,opacity:0}]);
        }
      } else {
        const last=make('polygon',{points:points(coordinates(s).output),fill:'#a64ca6',stroke:'#000','stroke-width':'.24'});
        outputLayer.append(last);
        animate(last,[{opacity:0},{opacity:1}]);
      }
      const clock=animate(particles,[{opacity:1},{opacity:1}]);
      if(s.phase==='move') {
        const from=coordinates(before),to=coordinates(s);
        const mix=(a,b,k)=>({x:a.x+(b.x-a.x)*k,y:a.y+(b.y-a.y)*k});
        motionTick=()=>{
          if(!moving||clock.playState==='paused'||clock.playState==='idle')return;
          const fraction=Math.min(1,Number(clock.currentTime||0)/duration), k=(1-Math.cos(Math.PI*fraction))/2;
          applyCoordinates({shift:mix(from.shift,to.shift,k),window:from.window.map((p,i)=>mix(p,to.window[i],k)),output:from.output.map((p,i)=>mix(p,to.output[i],k))});
          motionFrame=requestAnimationFrame(motionTick);
        };
        motionTick();
      }
      clock.finished.then(finish).catch(()=>{});
    }
    function play() {
      if(running){pause();return;}
      running=true;
      if(moving){effects.forEach(e=>e.play());motionTick?.();paint();return;}
      if(index===states.length-1)seek(0);
      advance(true);
    }
    buttons.next.addEventListener('click',()=>advance()); buttons.back.addEventListener('click',()=>seek(Math.max(0,index-1)));
    buttons.restart.addEventListener('click',()=>seek(0)); buttons.play.addEventListener('click',play);
    menu.addEventListener('change',()=>seek(Number(menu.value)));
    outer.addEventListener('keydown',event=>{
      if(event.repeat||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||event.target.closest('input,select,textarea,[contenteditable]'))return;
      const key=event.key.toLowerCase();if(!['n','p','k','r'].includes(key))return;
      event.preventDefault();if(key==='n')advance();else if(key==='p')seek(Math.max(0,index-1));else if(key==='r')seek(0);else play();
    });
    figure.addEventListener('mlsys-view-change',e=>{if(e.detail.view!=='explore')pause();else requestAnimationFrame(()=>{position();outer.scrollIntoView({block:'start'});});});
    document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
    new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)pause();}).observe(outer);
    media.addEventListener('change',()=>{if(media.matches&&moving){running=false;finishAction?.();}});
    staticNodes.forEach(n=>diagram.append(n)); figure.insertBefore(diagram,caption);figure.insertBefore(panel,caption);
    window.MLSysFigureViews.enhance({figure,diagram,explore:panel,initialView:new URLSearchParams(location.search).get('view')==='explore'?'explore':'diagram'});
    const position=()=>{
      const header=document.querySelector('#quarto-header');
      outer.style.setProperty('--life-header-offset',`${Math.max(0,header?.getBoundingClientRect().bottom||0)+16}px`);
    };
    window.addEventListener('scroll',position,{passive:true});
    position();if(document.querySelector('#quarto-header'))new ResizeObserver(position).observe(document.querySelector('#quarto-header'));
    const land=()=>{if(location.hash==='#fig-cnn')outer.scrollIntoView({block:'start'});};
    if(document.readyState==='complete')land();else window.addEventListener('load',land,{once:true});
    paint();outer.dataset.originalLifeReady='true';
  }
  document.querySelectorAll('[data-original-convolution]').forEach(outer=>{
    try{start(outer);}catch(error){console.error('[original-figure-preview]',error);}
  });
})();
