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
    const originalLinks=links.map(p=>({d:p.getAttribute('d'),transform:p.getAttribute('transform')}));
    const originalOutput=polygon(paths.find(p=>p.getAttribute('fill')==='rgb(64.99939%, 29.998779%, 64.99939%)'));
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
    svg.append(outputLayer,particles);
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
    const keys=node('p','Keyboard: N Next · P Back · K Play/Pause · R Restart. Focus the animation to use shortcuts.','life-keys');
    const legend=node('div','','life-legend');
    for (const [text,color] of [['Input window','#ff8000'],['Input contributions','#2455d6'],['Output activations','#a64ca6']]) {
      const chip=node('span',text); chip.style.setProperty('--chip',color); legend.append(chip);
    }
    const progress=node('p','','life-progress');
    panel.append(controls,keys,label,narration,canvas,legend,progress);
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
    let index=0, completed=0, running=false, moving=false, timer=null, effects=[], finishAction=null;
    const coordinates=s => {
      if(s.pose) return {window:[orangeOrigin,add(orangeOrigin,scale(lower.x,3)),add(orangeOrigin,scale(add(lower.x,lower.y),3)),add(orangeOrigin,scale(lower.y,3))],output:originalOutput,shift:{x:0,y:0}};
      const {row,col}=s, window=corners(lower,col,row,3);
      return {window,output:corners(upper,col,row),shift:minus(window[0],orangeOrigin)};
    };
    function applyPose(s) {
      const c=coordinates(s);
      if(s.pose){windowGroup.removeAttribute('transform');links.forEach((p,i)=>{p.setAttribute('d',originalLinks[i].d);p.setAttribute('transform',originalLinks[i].transform);});return;}
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
      progress.textContent=s.pose ? 'One input window → one output activation. No numerical values are specified in this figure.' : `${count} of 25 output activations stored · overlapping windows reuse input values.`;
      panel.dataset.outputs=String(count);
    }
    function paint() {
      const s=states[index]; if(!moving){applyPose(s);drawOutputs(s);}
      heading.textContent=s.title; explanation.textContent=s.text; menu.value=String(index);
      buttons.back.disabled=index===0; buttons.restart.disabled=index===0&&!moving;
      buttons.next.disabled=index===states.length-1&&!moving;
      buttons.next.textContent=moving?'Finish':'Next'; buttons.play.textContent=running?'Pause':index===states.length-1?'Replay':'Play';
      Object.assign(panel.dataset,{step:String(index),committed:String(completed),moving:String(moving),running:String(running)});
    }
    function cancel() {
      clearTimeout(timer); timer=null; effects.forEach(e=>e.cancel()); effects=[]; particles.replaceChildren(); finishAction=null;
    }
    function pause() {
      clearTimeout(timer);timer=null;running=false;effects.forEach(e=>e.pause());paint();
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
        const a=coordinates(before), b=coordinates(s);
        animate(windowGroup,[{transform:`translate(${a.shift.x}px,${a.shift.y}px)`},{transform:`translate(${b.shift.x}px,${b.shift.y}px)`}]);
        // Interpolate the connecting lines with the same clock as the window.
        links.forEach((link,i)=>animate(link,[{d:`path('M ${a.window[i].x} ${a.window[i].y} L ${a.output[i].x} ${a.output[i].y}')`},{d:`path('M ${b.window[i].x} ${b.window[i].y} L ${b.output[i].x} ${b.output[i].y}')`}]));
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
      clock.finished.then(finish).catch(()=>{});
    }
    function play() {
      if(running){pause();return;}
      running=true;
      if(moving){effects.forEach(e=>e.play());paint();return;}
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
