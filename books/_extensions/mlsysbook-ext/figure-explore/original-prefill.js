/* The chapter's verified TikZ artwork, with token identity and cache lifetime. */
(() => {
  'use strict';
  const NS='http://www.w3.org/2000/svg';
  const make=(tag,attrs={},text)=>{const el=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));if(text!==undefined)el.textContent=text;return el;};
  const lesson=[
    ['Start with the prompt','The four blue tokens are known. The cache is empty; future generated tokens have not appeared.','ready',0],
    ['Process the whole prompt','All four prompt tokens enter the same prefill pass. The same model weights serve every prompt position.','prefill',0],
    ['Save the prompt keys and values','Prefill writes one key/value entry for each prompt token. These entries remain available to later decode steps.','write-prompt',4],
    ['Predict the first token: the','The last prompt position predicts “the.” That prediction becomes the first decode input; its keys and values are not in the cache yet.','first',4],
    ['Decode the: read earlier entries','The input “the” enters the first narrow pass. Its attention reads the four prompt entries rather than recomputing them.','read',4,0],
    ['Append the consumed token: the','The pass saves the keys and values of its input, “the.” The cache now retains five consumed tokens.','append',5,0],
    ['Predict the next token: mat','The pass predicts “mat.” The cache ends at “the”; the newly predicted “mat” has not been processed yet.','emit',5,0],
    ['Feed mat into the next pass','The same “mat” prediction follows the dashed arrow into the next input position. No cache entry is added by this handoff.','handoff',5,1],
    ['Decode mat: read earlier entries','The input “mat” reads all five earlier key/value entries. One new input is processed while the stored context is reused.','read',5,1],
    ['Append the consumed token: mat','The pass saves the keys and values of “mat.” Earlier cache entries remain unchanged.','append',6,1],
    ['Predict the next token: today','The pass predicts “today.” It remains a prediction until the next decode pass consumes it.','emit',6,1],
    ['Feed today into the next pass','The same “today” prediction becomes the next input. The cache still ends at “mat.”','handoff',6,2],
    ['Decode today: read earlier entries','The input “today” reads all six earlier entries. The context read grows even though each pass receives only one new token.','read',6,2],
    ['Append the consumed token: today','The pass saves the keys and values of “today,” leaving seven retained entries.','append',7,2],
    ['Predict the end token','The pass emits [end], so generation stops. [end] is an output; no further pass consumes it and no [end] cache entry is written.','emit',7,2],
    ['Compare the two phases','Prefill shares the weight reads across the prompt. Decode repeats weight reads and reads a growing cache for each new input. This drawing illustrates the dependency, not measured latency.','complete',7,2],
    ['Release the request cache','Generation is complete and no attention step needs these entries. This request’s private cache is eligible for reclamation. Clearing the drawing does not imply physical zeroization or immediate deallocation, and does not describe shared-prefix eviction.','release',0,2]
  ];
  window.MLSysFigureLessons=window.MLSysFigureLessons||{};
  window.MLSysFigureLessons['original-prefill']=async(root,{outer,verifiedArtwork})=>{
    if(!verifiedArtwork||!window.MLSysOriginalPlayer)throw new Error('Original prefill artwork and shared player are required.');
    root.classList.add('mlsys-original-prefill');
    const svg=verifiedArtwork.cloneNode(true);svg.removeAttribute('width');svg.removeAttribute('height');
    svg.setAttribute('role','img');svg.setAttribute('aria-label','Original prefill and decode diagram: prompt tokens, model passes, generated tokens, and retained key/value entries.');
    const originalBox=svg.getAttribute('viewBox');
    // Keep glyph and clip references private to this instance of the original SVG.
    const ids=new Map();svg.querySelectorAll('[id]').forEach((el,i)=>{const old=el.id,id=`${outer.id}-prefill-art-${i}`;ids.set(old,id);el.id=id;});
    svg.querySelectorAll('*').forEach(el=>{for(const attr of [...el.attributes]){let value=attr.value;ids.forEach((id,old)=>{if(value===`#${old}`)value=`#${id}`;else value=value.replaceAll(`url(#${old})`,`url(#${id})`);});if(value!==attr.value)el.setAttributeNS(attr.namespaceURI,attr.name,value);}});
    const paths=[...svg.querySelectorAll('path')];
    const rectangles=paths.filter(p=>p.getAttribute('d')?.includes(' Z M ')&&p.hasAttribute('stroke'));
    const green=paths.filter(p=>p.getAttribute('fill')==='rgb(85.882568%, 99.215698%, 65.098572%)');
    const cache=rectangles.filter(p=>p.getAttribute('fill')==='rgb(98.039246%, 83.137512%, 68.624878%)');
    const passes=rectangles.filter(p=>p.getAttribute('fill')==='rgb(100%, 100%, 100%)');
    const prompt=rectangles.filter(p=>p.getAttribute('fill')==='rgb(81.959534%, 95.292664%, 100%)');
    if(green.length!==6||cache.length!==7||passes.length!==4||prompt.length!==4)throw new Error('Prefill artwork changed: review the object mapping.');
    const bounds=path=>{const d=path.getAttribute('d').match(/-?\d*\.?\d+(?:e[+-]?\d+)?/gi).map(Number),m=new DOMMatrix(path.getAttribute('transform')||undefined);const pts=[0,2,4,6].map(i=>new DOMPoint(d[i],d[i+1]).matrixTransform(m));return{x:Math.min(...pts.map(p=>p.x)),y:Math.min(...pts.map(p=>p.y)),w:Math.max(...pts.map(p=>p.x))-Math.min(...pts.map(p=>p.x)),h:Math.max(...pts.map(p=>p.y))-Math.min(...pts.map(p=>p.y))};};
    const gb=green.map(bounds),cb=cache.map(bounds),pb=passes.map(bounds),ib=prompt.map(bounds);
    const center=b=>({x:b.x+b.w/2,y:b.y+b.h/2});
    const layer=make('g',{'data-prefill-overlay':''}),motion=make('g',{'aria-hidden':'true'});svg.append(layer,motion);
    const tokens=['the','mat','mat','today','today','[end]'],retained=['The','cat','sat','on','the','mat','today'];
    const tokenLabels=gb.map((b,i)=>{const g=make('g',{'data-token':tokens[i]});g.append(make('rect',{x:b.x+.5,y:b.y+.5,width:b.w-1,height:b.h-1,fill:'#dcfda6'}));const label=make('text',{x:center(b).x,y:center(b).y,'dominant-baseline':'middle','text-anchor':'middle',fill:'#151515','font-size':6.5,'font-family':'sans-serif'},tokens[i]);g.append(label);layer.append(g);return{g,label};});
    const cacheLabels=cb.map((b,i)=>{const label=make('text',{x:center(b).x,y:center(b).y,'dominant-baseline':'middle','text-anchor':'middle',fill:'#5c2b00','font-size':i===6?3.1:3.6,'font-family':'sans-serif','data-cache-token':retained[i]},retained[i]);layer.append(label);return label;});
    const focus=make('rect',{fill:'none',stroke:'#005f8e','stroke-width':1.3,rx:1,'data-active-pass':''});layer.append(focus);
    const viewControl=document.createElement('label');viewControl.className='prefill-view-control';viewControl.textContent='Diagram view ';const view=document.createElement('select');view.setAttribute('aria-label','Diagram view');for(const [v,t]of [['follow','Follow the current phase'],['whole','Whole diagram']]){const o=document.createElement('option');o.value=v;o.textContent=t;view.append(o);}view.value=matchMedia('(max-width: 600px)').matches?'follow':'whole';viewControl.append(view);
    const inset=document.createElement('section');inset.className='prefill-cache-inset';inset.setAttribute('aria-label','Detail of the same retained KV cache');
    const insetTitle=document.createElement('strong');insetTitle.textContent='Cache detail — the same retained entries';
    const insetCells=document.createElement('div');insetCells.className='prefill-cache-cells';
    const insetNodes=retained.map(token=>{const n=document.createElement('span');n.textContent=token;insetCells.append(n);return n;});
    const insetOperation=document.createElement('p');inset.append(insetTitle,insetCells,insetOperation);
    const cacheStatus=document.createElement('p');cacheStatus.className='prefill-cache-status';cacheStatus.setAttribute('aria-live','off');
    let current=0,stage;
    function updateView(){inset.hidden=!(view.value==='follow'&&current>=3);if(view.value==='whole')svg.setAttribute('viewBox',originalBox);else svg.setAttribute('viewBox',current<3?'0 0 182 162.212':'178 0 185.372 162.212');}
    view.addEventListener('change',updateView);
    function render(index,context){
      current=index;stage=context.stage;
      if(!stage.contains(svg)){stage.append(viewControl,svg,inset,cacheStatus);}
      const [,,kind,count,k=0]=lesson[index];
      tokenLabels.forEach(({g,label},i)=>{let visible=false;if(i===0)visible=index>=3;else if(i===1)visible=index>=6;else if(i===2)visible=index>=7;else if(i===3)visible=index>=10;else if(i===4)visible=index>=11;else if(i===5)visible=index>=14;label.style.visibility=visible?'visible':'hidden';g.firstChild.setAttribute('fill',visible?'#dcfda6':'#fff');});
      cache.forEach((cell,i)=>{cell.style.fill=i<count?'rgb(98.039246%, 83.137512%, 68.624878%)':'white';cacheLabels[i].style.visibility=i<count?'visible':'hidden';cacheLabels[i].style.opacity='1';cell.style.strokeWidth=(kind==='read'&&i<count)?'1.4':'.59776';});
      insetNodes.forEach((n,i)=>{n.dataset.retained=String(i<count);n.classList.toggle('is-read',kind==='read'&&i<count);n.textContent=i<count?retained[i]:'·';});
      insetOperation.textContent=kind==='release'?'The request has finished. Its private cache can be reclaimed; generated tokens remain visible.':kind==='read'?`Read all ${count} earlier entries into the highlighted pass. These entries remain stored.`:kind==='append'?`Append the consumed input “${retained[count-1]}.” Earlier entries stay unchanged.`:'The newly predicted output is not a cache entry until a later pass consumes it.';
      const active=index>0&&index<15?(index<4?pb[0]:pb[k+1]):null;
      focus.style.visibility=active?'visible':'hidden';if(active)for(const [key,value]of Object.entries({x:active.x-1,y:active.y-1,width:active.w+2,height:active.h+2}))focus.setAttribute(key,value);
      cacheStatus.textContent=kind==='release'?'Request complete: no retained entries are needed by this request. Its private cache is eligible for reclamation.':count?`Retained KV entries (${count}): ${retained.slice(0,count).join(' · ')}. Each labeled cell stands for that token’s keys and values across the model.`:'Retained KV entries: none. Orange cells mark available places in this illustrative request cache.';
      root.dataset.prefillState=String(index);root.dataset.cacheCount=String(count);updateView();motion.replaceChildren();
    }
    const trajectory=(from,to,bend=0)=>t=>({x:from.x+(to.x-from.x)*t,y:from.y+(to.y-from.y)*t-bend*4*t*(1-t)});
    function pathsFor(index){const [,,kind,count,k=0]=lesson[index],block=pb[k+1];
      if(kind==='prefill')return ib.map(b=>trajectory({x:center(b).x,y:b.y+b.h},{x:center(b).x,y:pb[0].y}));
      if(kind==='write-prompt')return cb.slice(0,4).map(b=>trajectory({x:center(b).x,y:pb[0].y+pb[0].h},{x:center(b).x,y:b.y}));
      if(kind==='first')return[trajectory({x:pb[0].x+pb[0].w,y:center(pb[0]).y},{x:gb[0].x,y:center(gb[0]).y},-3)];
      if(kind==='read')return cb.slice(0,count).map((b,i)=>trajectory(center(b),{x:block.x+block.w*(i+1)/(count+1),y:block.y+block.h},12));
      if(kind==='append')return[trajectory({x:center(block).x,y:block.y+block.h},center(cb[count-1]),-8)];
      if(kind==='emit')return[trajectory({x:block.x+block.w,y:block.y},center(gb[k*2+1]))];
      if(kind==='handoff')return[trajectory({x:center(gb[k*2-1]).x,y:gb[k*2-1].y},{x:center(gb[k*2]).x,y:gb[k*2].y},8)];
      return[];
    }
    const controller=window.MLSysOriginalPlayer.create({root,outer,title:'How does a prompt become a growing KV cache?',scope:'One request. Words stand for tokens; cache cells stand for their keys and values. Attention includes the current input’s newly computed keys and values; arrows emphasize earlier entries. Append marks retention for later passes. Motion is illustrative.',steps:lesson.map(([label,description])=>({label,description})),render,transition:async({to,onFrame,signal})=>{
      const tracks=pathsFor(to);
      if(lesson[to][2]==='release'){
        onFrame(t=>cacheLabels.forEach(el=>el.style.opacity=String(1-t)));
        return;
      }
      if(!tracks.length)return;
      // Show the destination phase while its logical contributions move.
      if(view.value==='follow'&&to>=3){svg.setAttribute('viewBox','178 0 185.372 162.212');inset.hidden=false;}
      const nextKind=lesson[to][2],targetBlock=to<4?pb[0]:pb[(lesson[to][4]||0)+1];
      focus.style.visibility='visible';for(const [key,value]of Object.entries({x:targetBlock.x-1,y:targetBlock.y-1,width:targetBlock.w+2,height:targetBlock.h+2}))focus.setAttribute(key,value);insetOperation.textContent=nextKind==='read'?'Reading the retained entries into the highlighted pass…':nextKind==='append'?'Saving the consumed input’s keys and values…':insetOperation.textContent;
      if(nextKind==='read')insetNodes.forEach((n,i)=>n.classList.toggle('is-read',i<lesson[to][3]));
      const kind=lesson[to][2],color=['read','append','write-prompt'].includes(kind)?'#d76c00':kind==='prefill'?'#005f8e':'#008f45';
      const circles=tracks.map(()=>{const c=make('circle',{r:1.75,fill:color,stroke:'white','stroke-width':.45});motion.append(c);return c;});
      await onFrame(t=>tracks.forEach((track,i)=>{const p=track(t);circles[i].setAttribute('cx',p.x);circles[i].setAttribute('cy',p.y);}));
      if(signal.aborted)motion.replaceChildren();
    }});
    return controller;
  };
})();
