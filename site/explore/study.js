/* One workspace; authored collections and typed relationships, no inferred ontology. */
(async () => {
  'use strict';
  const root=document.getElementById('study-root'); if(!root)return;
  const $=id=>root.querySelector(`#study-${id}`);
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let data;
  try{const r=await fetch('framework-data.json',{cache:'no-cache'});if(!r.ok)throw Error();data=await r.json();}
  catch{$('loading').textContent='The field could not load. Definitions and static figures are available below.';return;}
  const f=data.framework, concepts=new Map(data.concepts.map(c=>[c.id,c]));
  const rows=f.areas.flatMap(a=>a.rows.map(r=>({...r,area:a}))), rowById=new Map(rows.map(r=>[r.id,r]));
  const homes=new Map();rows.forEach(r=>r.concepts.forEach(id=>homes.set(id,r)));
  const views=['tree','nested','connections'], primaryViews=['tree','connections'];
  let locateOnRender=false;
  let connectionScope='full', connectionRendered='', view='nested', area='', row='', concept='', query='', pan={x:0,y:0,z:1}, world={w:1440,h:1000}, drag=null, restoring=false, suppressNextClick=false, readerClosed=false;
  const expanded=new URLSearchParams(location.search).get('expanded')==='1';
  root.classList.toggle('study-expanded',expanded);
  if(expanded){$('expand').textContent='Return to page';$('expand').target='_self';}
  const short=o=>o.display_label||o.label;
  const members=r=>[...r.concepts,...(r.secondary_concepts||[])];
  const labelList=(r,key)=>(r[key]||[]).map(id=>f.dimensions[key].find(d=>d.id===id)?.label||id).join(' · ');
  const button=(type,id,label,extra='')=>`<button type="button" data-${type}="${esc(id)}" ${extra}>${esc(label)}</button>`;
  function save(push=false){
    if(restoring)return;
    const p=new URLSearchParams({view});if(view==='connections')p.set('scope',connectionScope); if(area)p.set('area',area);if(row)p.set('subject',row);if(concept)p.set('concept',concept);if(query)p.set('q',query);
    history[push?'pushState':'replaceState'](null,'',`#${p}`);
    const u=new URL(location.href);expanded?u.searchParams.delete('expanded'):u.searchParams.set('expanded','1');$('expand').href=u.href;
  }
  function transform(){$('world').style.transform=`translate(${pan.x}px,${pan.y}px) scale(${pan.z})`;}
  function fit(forceAll=false){const b=$('map').getBoundingClientRect(), phone=matchMedia('(max-width:700px)').matches,narrow=phone&&view!=='connections';pan.z=Math.max(.02,Math.min(view==='connections'?2.1:1.25,(b.width-32)/world.w,...(narrow?[]:[(b.height-40)/world.h])));pan.x=(b.width-world.w*pan.z)/2;pan.y=narrow?14:(b.height-world.h*pan.z)/2;if(phone&&view==='connections'&&connectionScope!=='full'&&!forceAll)pan.z=Math.max(.9,pan.z);transform();if(phone&&view==='connections'&&connectionScope!=='full'&&!forceAll){const selected=$('nodes').querySelector('.study-selected-concept');if(selected){const nb=selected.getBoundingClientRect();pan.x+=b.width/2-(nb.left-b.left+nb.width/2);pan.y+=b.height/2-(nb.top-b.top+nb.height/2);transform();}}}
  function zoom(factor,cx,cy){const b=$('map').getBoundingClientRect();cx??=b.width/2;cy??=b.height/2;const z=Math.max(.02,Math.min(4,pan.z*factor));pan.x=cx-(cx-pan.x)*z/pan.z;pan.y=cy-(cy-pan.y)*z/pan.z;pan.z=z;transform();}
  function locateSelected(){
    const ids=new Set([concept,...data.relationships.filter(e=>e.source===concept||e.target===concept).flatMap(e=>[e.source,e.target])]);
    const boxes=Array.from($('nodes').querySelectorAll('[data-concept]')).filter(n=>ids.has(n.dataset.concept)).map(n=>n.getBoundingClientRect());
    if(!boxes.length)return;
    const b=$('map').getBoundingClientRect(),left=Math.min(...boxes.map(n=>n.left)),right=Math.max(...boxes.map(n=>n.right)),top=Math.min(...boxes.map(n=>n.top)),bottom=Math.max(...boxes.map(n=>n.bottom));
    const x=(left-b.left+(right-left)/2-pan.x)/pan.z,y=(top-b.top+(bottom-top)/2-pan.y)/pan.z,w=(right-left)/pan.z,h=(bottom-top)/pan.z;
    pan.z=Math.max(.02,Math.min(1.2,(b.width-64)/w,(b.height-64)/h));pan.x=b.width/2-x*pan.z;pan.y=b.height/2-y*pan.z;transform();
  }
  function node(type,id,label,x,y,w,h,cls='',sub=''){
    const active=(type==='concept'&&id===concept)||(type==='row'&&id===row)||(type==='area'&&id===area);
    return `<button type="button" class="study-node study-node-${type} ${cls}${active?' study-node-selected':''}" data-${type}="${esc(id)}" style="left:${x}px;top:${y}px;width:${w}px;height:${h}px" aria-label="${esc(label)}"><span>${esc(label)}</span>${sub?`<small>${esc(sub)}</small>`:''}</button>`;
  }
  function line(x1,y1,x2,y2){const middle=(x1+x2)/2;return `<path d="M${x1},${y1} C${middle},${y1} ${middle},${y2} ${x2},${y2}"/>`;}
  function paint(nodes,edges,w,h){world={w,h};$('world').style.width=`${w}px`;$('world').style.height=`${h}px`;$('edges').setAttribute('width',w);$('edges').setAttribute('height',h);$('edges').innerHTML=edges;$('nodes').innerHTML=nodes;$('loading').hidden=true;}
  function tree(){
    let nodes='',edges='';
    if(matchMedia('(max-width:700px)').matches){
      const a=f.areas.find(a=>a.id===area),r=rowById.get(row),items=r?members(r).map(id=>({type:'concept',id,label:concepts.get(id)?.label||id})):a?a.rows.map(r=>({type:'row',id:r.id,label:short(r)})):f.areas.map(a=>({type:'area',id:a.id,label:short(a)}));
      let top=30;
      if(a){nodes+=node('area',a.id,short(a),70,top,375,64);top+=96;}
      else{nodes+=node('home','','ML systems',70,top,375,64,'study-node-root');top+=96;}
      if(r){nodes+=node('row',r.id,short(r),70,top,375,64);edges+=`<path d="M42,94 V${top+32} H70"/>`;top+=96;}
      items.forEach((item,index)=>{const y=top+index*78;nodes+=node(item.type,item.id,item.label,70,y,375,64);edges+=`<path d="M42,94 V${y+32} H70"/>`;});
      paint(nodes,edges,515,Math.max(580,top+items.length*78+24));return;
    }
    const h=850,rootY=h/2;
    nodes+=node('home','','ML systems',15,rootY-42,160,84,'study-node-root');
    f.areas.forEach((a,index)=>{
      const y=25+index*66,center=y+28;
      nodes+=node('area',a.id,short(a),225,y,250,56,'');edges+=line(175,rootY,225,center);
      if(area!==a.id)return;
      const rowStart=Math.max(12,Math.min(h-a.rows.length*48-12,center-a.rows.length*24));
      a.rows.forEach((r,j)=>{
        const ry=rowStart+j*48;nodes+=node('row',r.id,short(r),550,ry,285,40,r.concepts.length?'':'study-node-proposed');edges+=line(475,center,550,ry+20);
        if(row!==r.id)return;
        const ids=members(r),start=Math.max(10,Math.min(h-ids.length*61-10,ry-ids.length*30));
        ids.forEach((id,k)=>{const c=concepts.get(id);if(!c)return;const cy=start+k*61;nodes+=node('concept',id,c.label,915,cy,285,56);edges+=line(835,ry+20,915,cy+28);});
      });
    });paint(nodes,edges,row?1220:area?855:495,h);
  }
  function nested(){
    let nodes='',edges='';
    const selectedRow=rowById.get(row), selectedArea=f.areas.find(a=>a.id===area), narrow=matchMedia('(max-width:700px)').matches, panelW=narrow?515:1000;
    if(selectedRow){
      nodes+=node('area',selectedRow.area.id,short(selectedRow.area),30,20,panelW-60,64,'study-node-band',narrow?'':selectedRow.area.question);
      nodes+=node('row',selectedRow.id,selectedRow.label,30,110,panelW-60,72,'study-node-band',narrow?'':selectedRow.definition);
      const ids=members(selectedRow);const cols=narrow?1:2,w=455,gap=30;
      ids.forEach((id,index)=>{const c=concepts.get(id);if(!c)return;nodes+=node('concept',id,c.label,30+(index%cols)*(w+gap),225+Math.floor(index/cols)*106,w,88,'study-node-concept-card',narrow?'':c.definition);});
      if(!ids.length)nodes+=`<div class="study-world-note" style="left:50px;top:235px;width:900px">This proposed subject does not yet have concept records. Its scope and boundaries can be inspected in the detail panel.</div>`;
      paint(nodes,edges,panelW,Math.max(520,270+Math.ceil(ids.length/cols)*106));return;
    }
    if(selectedArea){
      nodes+=node('area',selectedArea.id,selectedArea.label,30,20,panelW-60,75,'study-node-band',narrow?'':selectedArea.question);
      selectedArea.rows.forEach((r,index)=>{
        const cols=narrow?1:2,x=30+(index%cols)*485,y=135+Math.floor(index/cols)*(narrow?125:190);
        nodes+=node('row',r.id,short(r),x,y,455,narrow?105:166,'study-node-subject-card',narrow?'':r.definition);
      });
      paint(nodes,edges,panelW,Math.max(560,160+Math.ceil(selectedArea.rows.length/(narrow?1:2))*(narrow?125:190)));return;
    }
    const cols=matchMedia('(max-width:700px)').matches?2:4,cellW=342,cellH=160,gap=17;
    f.areas.forEach((a,index)=>{
      const x=24+(index%cols)*(cellW+gap),y=25+Math.floor(index/cols)*(cellH+gap);
      nodes+=`<div class="study-area-frame" style="left:${x}px;top:${y}px;width:${cellW}px;height:${cellH}px"></div>`;
      nodes+=node('area',a.id,short(a),x+18,y+20,cellW-36,125,'study-node-area-overview',`${a.rows.length} subjects`);
    });paint(nodes,edges,cols*(cellW+gap)+34,Math.ceil(f.areas.length/cols)*(cellH+gap)+44);
  }

  let connectionRun=0;
  const connectionCache=new Map();
  async function connections(){
    const id=concept || rowById.get(row)?.concepts[0] || 'mlsys.quantization';
    const key=connectionScope==='full'?'all':id, preserve=connectionRendered===key;
    const run=++connectionRun;
    if(!connectionCache.has(key)){
      $('loading').hidden=false;$('loading').textContent='Drawing the connections…';
      try{const response=await fetch(key==='all'?'field-all.svg':`field-focus/${encodeURIComponent(id)}.svg`,{cache:'no-cache'});if(!response.ok)throw Error();connectionCache.set(key,await response.text());}
      catch{if(run!==connectionRun)return;paint('', '',800,400);$('loading').hidden=false;$('loading').textContent='A connection drawing has not been generated for this concept yet. Its relationships are available in the detail panel.';return;}
    }
    if(run!==connectionRun || view!=='connections')return;
    const parsed=new DOMParser().parseFromString(connectionCache.get(key),'image/svg+xml');
    const svg=parsed.documentElement;
    if(svg.localName!=='svg')return;
    svg.querySelectorAll('script,foreignObject').forEach(n=>n.remove());
    svg.querySelectorAll('*').forEach(n=>Array.from(n.attributes).forEach(a=>{if(a.name.startsWith('on')||a.localName==='href')n.removeAttribute(a.name);}));
    svg.querySelectorAll('g.node[id^="concept-"]').forEach(n=>{const cid=n.id.slice(8);n.dataset.concept=cid;n.setAttribute('tabindex','0');n.setAttribute('role','button');n.setAttribute('aria-label',`${concepts.get(cid)?.label||cid}${homes.has(cid)?', collected under '+short(homes.get(cid)):''}`);n.classList.toggle('study-selected-concept',cid===concept);const near=new Set([concept,...data.relationships.filter(e=>e.source===concept||e.target===concept).flatMap(e=>[e.source,e.target])]);n.classList.toggle('study-context-dim',Boolean(concept)&&!near.has(cid));n.classList.toggle('study-neighbor',Boolean(concept)&&near.has(cid));});
    svg.querySelectorAll('g.edge[id^="relation-"]').forEach(n=>{const rid=n.id.slice(9),edge=data.relationships.find(e=>e.id===rid);n.dataset.relation=rid;n.classList.toggle('study-context-dim',Boolean(concept)&&edge?.source!==concept&&edge?.target!==concept);n.setAttribute('tabindex','0');n.setAttribute('role','button');n.setAttribute('aria-label',edge?`${concepts.get(edge.source)?.label} ${edge.predicate} ${concepts.get(edge.target)?.label}. Inspect conditions.`:'Inspect relationship');});
    const box=svg.viewBox.baseVal,w=box.width||800,h=box.height||500;
    svg.setAttribute('width',w);svg.setAttribute('height',h);svg.classList.add('study-connection-svg');
    paint('', '',w,h);$('nodes').appendChild(document.importNode(svg,true));if(!preserve)fit(true);else transform();connectionRendered=key;if(locateOnRender){requestAnimationFrame(locateSelected);locateOnRender=false;}
  }
  function breadcrumb(){
    let content=button('home','','All subjects');
    const a=f.areas.find(a=>a.id===area),r=rowById.get(row),c=concepts.get(concept);
    if(a)content+=`<span aria-hidden="true">/</span>${button('area',a.id,short(a))}`;
    if(r)content+=`<span aria-hidden="true">/</span>${button('row',r.id,short(r))}`;
    if(c)content+=`<span aria-hidden="true">/</span><span aria-current="page">${esc(c.label)}</span>`;
    $('breadcrumb').innerHTML=content;
  }
  function renderMap(refit=true){
    primaryViews.forEach(v=>{const active=v==='tree'?view!=='connections':view==='connections';$('tab-'+v).setAttribute('aria-selected',String(active));$('tab-'+v).tabIndex=active?0:-1;});
    $('map').setAttribute('aria-labelledby',`study-tab-${view==='connections'?'connections':'tree'}`);
    root.classList.toggle('study-layout-tree',view==='tree');$('taxonomy-layout').hidden=view==='connections';if(view!=='connections')$('taxonomy-layout').value=view;$('connection-scope').hidden=view!=='connections';$('connection-scope').value=connectionScope;if(view!=='connections')connectionRendered='';
    ({tree,nested,connections})[view]();breadcrumb();
    $('status').textContent=view==='connections'?`${connectionScope==='full'?`Full map · ${concepts.size} concepts · ${data.relationships.length} authored relationships. Zoom in to read labels.`:'Neighborhood · selected concept and its direct connections.'} Select a concept or arrow to inspect it.`:view==='tree'?'Select a subject to explore its concepts.':`${f.areas.length} subject areas · ${rows.length} subjects · ${concepts.size} concepts · Draft`;
    if(refit&&view!=='connections')requestAnimationFrame(()=>fit());
  }
  function books(ids){const links=data.book_links.filter(l=>ids.includes(l.concept));const unique=[...new Map(links.map(l=>[l.url,l])).values()];return `<details class="study-book-disclosure"><summary>Read in the book${unique.length?` · ${unique.length} ${unique.length===1?'passage':'passages'}`:''}</summary>${unique.length?`<ul>${unique.map(l=>`<li><a href="${esc(l.url)}"><span>${esc(l.volume?.replace('vol','Volume '))} · ${esc(l.chapter_title)}</span>${esc(l.label||concepts.get(l.concept)?.label||l.book_concept)} ↗</a></li>`).join('')}</ul><p>Candidate related passages; teaching depth has not been assessed.</p>`:'<p>No passage is linked yet. This does not establish whether the books cover this idea.</p>'}</details>`;}
  function related(c){
    const edges=(data.relationships||[]).filter(e=>e.source===c.id||e.target===c.id);
    if(!edges.length)return '<p class="study-muted">No relationships have been authored for this concept yet.</p>';
    return `<ul class="study-relations">${edges.map(e=>{
      const target=e.source===c.id?e.target:e.source,other=concepts.get(target);if(!other)return '';
      const sentence=e.source===c.id?`<span class="study-relation-subject">${esc(c.label)}</span><strong>${esc(e.predicate)}</strong>${button('concept',target,other.label)}`:`${button('concept',target,other.label)}<strong>${esc(e.predicate)}</strong><span class="study-relation-subject">${esc(c.label)}</span>`;
      return `<li>${sentence}${e.conditions?`<details><summary>When this applies</summary><p>${esc(e.conditions)}</p></details>`:''}</li>`;
    }).join('')}</ul>`;
  }
  function reader(){
    const c=concepts.get(concept),r=rowById.get(row),a=f.areas.find(a=>a.id===area);
    $('reader').hidden=false;root.classList.toggle('study-has-reader',!readerClosed&&Boolean(c||r||a));
    if(readerClosed||(!c&&!r&&!a)){$('reader').hidden=true;return;}
    $('close').hidden=false;
    if(!c&&!r&&a){$('reader-kind').textContent='AREA';$('detail').innerHTML=`<h2 id="study-reader-title">${esc(short(a))}</h2><p class="study-definition">${esc(a.question)}</p><p>${esc(a.scope)}</p>`;return;}
    if(c){
      $('reader-kind').textContent='CONCEPT';
      $('detail').innerHTML=`<h2 id="study-reader-title" tabindex="-1">${esc(c.label)}</h2><p class="study-definition">${esc(c.definition)}</p>${view==='connections'?'<button class="study-reader-action" type="button" data-locate>Locate on map</button>':'<button class="study-reader-action" type="button" data-open-map>View connections</button>'}<h3>Connections</h3>${related(c)}${books([c.id])}`;
    }else{
      $('reader-kind').textContent='SUBJECT';
      const ids=members(r),text=x=>Array.isArray(x)?x.join(' '):x;
      $('detail').innerHTML=`<h2 id="study-reader-title" tabindex="-1">${esc(short(r))}</h2><p class="study-definition">${esc(r.definition)}</p>${!ids.length?'<p class="study-muted">Concept records have not yet been authored for this proposed subject.</p>':''}<details class="study-boundaries"><summary>Scope and perspectives</summary>${r.includes?`<h3>Includes</h3><p>${esc(text(r.includes))}</p>`:''}${r.excludes?`<h3>Boundary</h3><p>${esc(text(r.excludes))}</p>`:''}<h3>Activities</h3><p>${esc(labelList(r,'activities'))}</p><h3>System perspectives</h3><p>${esc(labelList(r,'layers'))}</p><h3>Concerns</h3><p>${esc(labelList(r,'concerns'))}</p></details>${books(ids)}`;
    }
  }
  function select(type,id,push=true){
    readerClosed=false;query='';$('search').value='';$('search-results').hidden=true;
    if(type==='home'){area='';row='';concept='';if(view==='connections'){connectionScope='full';connectionRendered='';}}
    if(type==='area'){if(!f.areas.some(a=>a.id===id))return;area=id;row='';concept='';}
    if(type==='row'){const r=rowById.get(id);if(!r)return;area=r.area.id;row=id;concept='';}
    if(type==='concept'){const c=concepts.get(id);if(!c)return;concept=id;const home=homes.get(id);if(home){row=home.id;area=home.area.id;}}
    reader();renderMap(type==='home'||view!=='tree');save(push);
  }
  function search(){
    const q=query.trim().toLowerCase();
    if(!q){$('search-results').hidden=true;return;}
    const areas=f.areas.filter(a=>`${a.label} ${a.scope}`.toLowerCase().includes(q));
    const subjectRows=rows.filter(r=>`${r.label} ${r.definition}`.toLowerCase().includes(q));
    const matches=data.concepts.filter(c=>`${c.label} ${c.definition}`.toLowerCase().includes(q));
    $('search-results').hidden=false;
    $('search-results').innerHTML=`<h2>Find your place</h2>${areas.length?`<h3>Areas</h3>${areas.map(a=>button('area',a.id,a.label)).join('')}`:''}${subjectRows.length?`<h3>Subjects</h3>${subjectRows.map(r=>button('row',r.id,r.label)).join('')}`:''}${matches.length?`<h3>Concepts</h3>${matches.map(c=>button('concept',c.id,c.label)).join('')}`:''}${!areas.length&&!subjectRows.length&&!matches.length?'<p>No matching subject or concept.</p>':''}`;
  }
  root.addEventListener('click',event=>{if(event.target.closest('[data-locate]')){locateSelected();return;}if(event.target.closest('[data-open-map]')){locateOnRender=true;setView('connections');return;}if(suppressNextClick){suppressNextClick=false;event.preventDefault();event.stopPropagation();return;}const relation=event.target.closest('[data-relation]');if(relation){showRelation(relation.dataset.relation);return;}for(const type of ['home','area','row','concept']){const b=event.target.closest(`[data-${type}]`);if(b){if(type==='concept'&&event.target.closest('#study-search-results'))locateOnRender=true;select(type,b.dataset[type]);return;}}});
  function setView(next){view=next==='tree'?$('taxonomy-layout').value:next;readerClosed=false;reader();renderMap();save(true);}
  $('connection-scope').addEventListener('change',()=>{connectionScope=$('connection-scope').value;connectionRendered='';if(connectionScope==='local'&&!concept)select('concept','mlsys.quantization',false);else renderMap();save(true);});
  $('taxonomy-layout').addEventListener('change',()=>{view=$('taxonomy-layout').value;reader();renderMap();save(true);});
  primaryViews.forEach(v=>{$('tab-'+v).addEventListener('click',()=>setView(v));$('tab-'+v).addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const i=primaryViews.indexOf(v),next=event.key==='Home'?0:event.key==='End'?1:(i+1)%2;setView(primaryViews[next]);$('tab-'+primaryViews[next]).focus();});});
  function showRelation(id){
    readerClosed=false;const e=data.relationships.find(e=>e.id===id);if(!e)return;
    root.classList.add('study-has-reader');$('reader').hidden=false;$('close').hidden=false;$('reader-kind').textContent='RELATIONSHIP';
    $('detail').innerHTML=`<h2 id="study-reader-title">How these ideas connect</h2><p class="study-exact-relation">${button('concept',e.source,concepts.get(e.source)?.label||e.source)} <strong>${esc(e.predicate)}</strong> ${button('concept',e.target,concepts.get(e.target)?.label||e.target)}</p><h3>When this applies</h3><p class="study-definition">${esc(e.conditions||'Conditions have not yet been authored.')}</p>${books([e.source,e.target])}`;
  }
  root.addEventListener('keydown',event=>{const n=event.target.closest('g[data-concept],g[data-relation]');if(n&&['Enter',' '].includes(event.key)){event.preventDefault();n.dataset.concept?select('concept',n.dataset.concept):showRelation(n.dataset.relation);}if(event.key==='Escape'&&!query){$('close').click();}});
  $('close').addEventListener('click',()=>{readerClosed=true;reader();const id=concept||row||area,type=concept?'concept':row?'row':'area';requestAnimationFrame(()=>$('nodes').querySelector(`[data-${type}="${CSS.escape(id)}"]`)?.focus({preventScroll:true}));});
  $('search').addEventListener('input',()=>{query=$('search').value;search();save();});
  $('search').addEventListener('keydown',event=>{if(event.key==='Escape'){event.stopPropagation();query='';$('search').value='';search();save();}if(event.key==='ArrowDown'&&!$('search-results').hidden){event.preventDefault();$('search-results').querySelector('button')?.focus();}});
  $('overview').addEventListener('click',()=>select('home',''));$('plus').addEventListener('click',()=>zoom(1.25));$('minus').addEventListener('click',()=>zoom(.8));$('fit').addEventListener('click',()=>fit(true));
  const map=$('map');
  map.addEventListener('wheel',event=>{if(!event.ctrlKey&&!event.metaKey)return;event.preventDefault();const b=map.getBoundingClientRect();zoom(event.deltaY<0?1.12:1/1.12,event.clientX-b.left,event.clientY-b.top);},{passive:false});
  map.addEventListener('pointerdown',event=>{if(event.target.closest('button,input,a,summary,[role=button]')||event.button!==0)return;map.focus({preventScroll:true});drag={x:event.clientX,y:event.clientY,px:pan.x,py:pan.y};map.setPointerCapture(event.pointerId);map.classList.add('study-dragging');});
  map.addEventListener('pointermove',event=>{if(!drag||Math.hypot(event.clientX-drag.x,event.clientY-drag.y)<5)return;suppressNextClick=true;pan.x=drag.px+event.clientX-drag.x;pan.y=drag.py+event.clientY-drag.y;transform();});
  ['pointerup','pointercancel','lostpointercapture'].forEach(type=>map.addEventListener(type,()=>{drag=null;map.classList.remove('study-dragging');setTimeout(()=>{suppressNextClick=false;},0);}));
  map.addEventListener('keydown',event=>{if(event.target!==map)return;const actions={ArrowLeft:()=>pan.x+=50,ArrowRight:()=>pan.x-=50,ArrowUp:()=>pan.y+=50,ArrowDown:()=>pan.y-=50,'+':()=>zoom(1.25),'=':()=>zoom(1.25),'-':()=>zoom(.8),Home:()=>fit(true)};if(actions[event.key]){event.preventDefault();actions[event.key]();transform();}});
  function restore(){restoring=true;readerClosed=false;const p=new URLSearchParams(location.hash.slice(1));connectionScope=p.get('scope')==='local'?'local':'full';connectionRendered='';view=views.includes(p.get('view'))?p.get('view'):'nested';area=f.areas.some(a=>a.id===p.get('area'))?p.get('area'):'';row=rowById.has(p.get('subject'))?p.get('subject'):'';concept=concepts.has(p.get('concept'))?p.get('concept'):'';if(concept&&homes.has(concept)){row=homes.get(concept).id;area=homes.get(concept).area.id;}if(row)area=rowById.get(row).area.id;query=p.get('q')||'';$('search').value=query;reader();renderMap();search();restoring=false;save();}
  $('scope').textContent=f.scope||'';$('version').textContent=`${f.version} · ${f.status}. Source-driven draft; subject boundaries and relationships remain under review.`;
  let previousWidth=0,previousHeight=0;new ResizeObserver(entries=>{const {width,height}=entries[0].contentRect;if(Math.abs(width-previousWidth)>20||Math.abs(height-previousHeight)>20){previousWidth=width;previousHeight=height;fit();}}).observe(map);
  matchMedia('(max-width:700px)').addEventListener('change',()=>renderMap());
  window.addEventListener('popstate',restore);restore();
})();
