/* One subject model; progressive browsing, internal book links, and focused maps. */
(async () => {
  "use strict";
  const $ = id => document.getElementById(`field-${id}`);
  const escape = value => String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
  let model;
  try {
    if ($("data")) model=JSON.parse($("data").textContent);
    else { const r=await fetch("field-data.json"); if(!r.ok) throw Error(); model=await r.json(); }
  } catch { $("detail").innerHTML='<h2>The concepts could not load</h2><p>Try reloading, or use the written taxonomy draft below.</p>'; return; }
  const nodes=new Map(model.concepts.map(n=>[n.id,n]));
  const families=new Map(model.families.map(f=>[f.id,f]));
  $("topic-select").innerHTML+=model.families.map(f=>`<option value="${escape(f.id)}">${escape(f.label)}</option>`).join("");
  const canvas=$("canvas");
  let svg, selected=null, focusId=null, viewName="focus", drag=null, landscapeContext=null;
  let bookFilter="", restoring=false, activeFamily="", activeTab="text", mobileStep="topics";
  const visibleIds=()=>viewName==="focus"?model.neighborhoods[focusId]:model.views.find(v=>v.id===viewName).concepts;
  const element=id=>svg?.querySelector(`[id="concept-${CSS.escape(id)}"]`);
  const matchingLinks=id=>model.book_links.filter(l=>l.concept===id&&(!bookFilter||l.volume===bookFilter));
  let expanded=new URLSearchParams(location.hash.slice(1)).get("expanded")==="1";
  document.querySelector(".field-explorer").classList.toggle("field-expanded",expanded);
  if(expanded){const browse=new URL(location.href);const q=new URLSearchParams(browse.hash.slice(1));q.delete("expanded");browse.hash=q.toString();$("browse").href=browse.href;$("browse").hidden=false;$("browse").onclick=e=>{e.preventDefault();expanded=false;document.querySelector(".field-explorer").classList.remove("field-expanded");$("browse").hidden=true;save();};}
  if(location.pathname.endsWith("field-preview.html")) $("book-atlas").href="atlas.html";
  $("model-summary").textContent=`${model.summary.concepts} concepts · ${model.summary.relationships} draft relationships · ${model.summary.book_links} candidate passage links. Counts do not establish completeness.`;
  function save(push=false) {
    const p=new URLSearchParams();
    if(selected)p.set("concept",selected);if(activeFamily)p.set("topic",activeFamily);
    if(viewName!=="focus")p.set("view",viewName);if(bookFilter)p.set("book",bookFilter);
    if(activeTab!=="text")p.set("tab",activeTab);if(mobileStep!=="topics")p.set("step",mobileStep);
    if($("search").value)p.set("q",$("search").value);if(expanded)p.set("expanded","1");
    if(!restoring){
      const old=history.state||{};
      const state={...old,viewBox:svg?box():null,step:mobileStep,q:$("search").value,from:push?old.step:old.from};
      history[push?"pushState":"replaceState"](state,"",p.size?`#${p}`:location.pathname+location.search);
    }
    const target=new URL(location.href);p.set("expanded","1");p.set("tab","map");target.hash=p.toString();$("expand").href=target.href;
  }
  function setBox(box) { svg.setAttribute("viewBox", box.join(" ")); }
  function box() { const v = svg.viewBox.baseVal; return [v.x,v.y,v.width,v.height]; }
  function fit(ids) {
    const inverse = svg.getScreenCTM()?.inverse();
    if (!inverse) return;
    const points = ids.map(element).filter(Boolean).flatMap((el) => {
      const b = el.getBBox(), m = inverse.multiply(el.getScreenCTM());
      return [[b.x,b.y],[b.x+b.width,b.y+b.height]].map(([x,y]) => new DOMPoint(x,y).matrixTransform(m));
    });
    if (!points.length) return;
    const x = Math.min(...points.map((p) => p.x)), y = Math.min(...points.map((p) => p.y));
    const width = Math.max(...points.map((p) => p.x))-x, height = Math.max(...points.map((p) => p.y))-y;
    setBox([x-55,y-45,Math.max(220,width+110),Math.max(160,height+90)]);
  }
  function zoom(factor) {
    const [x,y,w,h] = box();
    if (w*factor < 150 || w*factor > 80000) return;
    setBox([x+w*(1-factor)/2,y+h*(1-factor)/2,w*factor,h*factor]);
  }
  function updateOverlay() {
    for(const n of model.concepts) element(n.id)?.classList.toggle("linked",matchingLinks(n.id).length>0);
    $("status").textContent=viewName==="focus"?`${nodes.get(focusId).label} and ${visibleIds().length-1} connected concepts`:(viewName==="all"?"Full draft landscape":"Selected connections");
  }
  function highlight() {
    const adjacent=new Set([selected]);
    model.relationships.filter(e=>e.source===selected||e.target===selected).forEach(e=>{adjacent.add(e.source);adjacent.add(e.target);});
    for(const n of model.concepts) {
      const el=element(n.id);if(!el)continue;
      el.classList.toggle("selected",n.id===selected);el.classList.toggle("muted",!adjacent.has(n.id));el.setAttribute("aria-pressed",String(n.id===selected));
    }
    for(const e of model.relationships) {
      const el=svg.querySelector(`[id="relation-${CSS.escape(e.id)}"]`);if(!el)continue;
      const active=e.source===selected||e.target===selected;
      el.classList.toggle("active",active);el.classList.toggle("muted",!active);
    }
    document.querySelectorAll("[data-concept]").forEach(b=>b.setAttribute("aria-current",b.dataset.concept===selected?"true":"false"));
  }
  function renderMap(view="focus") {
    viewName=view;focusId=view==="focus"?selected:null;
    const diagram=view==="focus"?`focus-${selected}`:view;
    const doc=new DOMParser().parseFromString(model.diagrams[diagram],"image/svg+xml");
    svg=document.importNode(doc.documentElement,true);canvas.replaceChildren(svg);
    document.querySelector(".field-map-region").hidden=false;
    $("view").querySelector('[value="focus"]').disabled=false;$("view").value=view;
    $("neighbors").disabled=view==="focus";$("neighbors").hidden=view==="focus";$("return").hidden=!landscapeContext||view!=="focus";
    $("download").href=view==="focus"?`field-focus/${selected}.svg`:`field-${view}.svg`;
    for(const n of model.concepts){const el=element(n.id);if(!el)continue;
      el.setAttribute("tabindex","0");el.setAttribute("role","button");el.setAttribute("aria-label",n.label);
      el.onclick=()=>navigate(n.id,false,true,true);el.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();navigate(n.id,true,true,true);}};
    }
    updateOverlay();highlight();fit(visibleIds());
  }
  function conceptButton(id) {
    const n=nodes.get(id);
    return `<button type="button" data-concept="${escape(id)}" aria-current="${id===selected}">${escape(n.label)}</button>`;
  }
  function setStep(step) { mobileStep=step;document.querySelector(".field-browser").dataset.step=step; }
  function renderNavigation() {
    $("topic-select").value=activeFamily;
    const query=$("search").value.trim().toLowerCase();
    $("clear-search").hidden=!query;
    let matches=[];
    if(query) {
      matches=model.concepts.filter(n=>`${n.label} ${n.definition} ${(n.aliases||[]).join(" ")}`.toLowerCase().includes(query));
      $("topic-title").textContent="Search results";$("topic-question").textContent=`For “${$("search").value.trim()}” across all topics`;
    } else if(activeFamily) {
      matches=model.concepts.filter(n=>n.family===activeFamily);
      $("topic-title").textContent=families.get(activeFamily).label;$("topic-question").textContent=families.get(activeFamily).question;
    } else {
      $("topic-title").textContent="Choose a topic";$("topic-question").textContent="Open a topic on the left to browse its concepts.";
    }
    $("results").textContent=query||activeFamily?`${matches.length} ${matches.length===1?"concept":"concepts"}`:"";
    $("list").innerHTML=matches.map(n=>`<div class="field-concept-row">${conceptButton(n.id)}<p>${escape(query?families.get(n.family).label:n.definition)}</p></div>`).join("")+(query&&!matches.length?'<p>No matching concepts. Try a shorter term or an abbreviation.</p>':"");
    $("taxonomy").querySelectorAll("[data-topic]").forEach(b=>b.setAttribute("aria-current",b.dataset.topic===activeFamily?"true":"false"));
  }
  function renderTopics() {
    const topics=model.families.map(f=>`<li><button type="button" data-topic="${escape(f.id)}">${escape(f.label)}<span>${model.concepts.filter(n=>n.family===f.id).length}</span></button></li>`).join("");
    const t=model.taxonomy;
    const branch=id=>{
      if(nodes.has(id))return `<li>${conceptButton(id)}</li>`;
      const topic=t.topics.find(x=>x.id===id);
      return `<li><details><summary>${escape(topic.label)}</summary><ul>${t.hierarchy.filter(e=>e.broader===id).map(e=>branch(e.narrower)).join("")}</ul></details></li>`;
    };
    $("taxonomy").innerHTML=`<h2>Topics</h2><ul class="field-topics">${topics}</ul>${t?`<div class="field-classified"><h2>Classified branches</h2><p>Broader and narrower subjects · pilot</p><ul>${t.roots.map(branch).join("")}</ul></div>`:""}`;
  }
  function chooseTopic(id,push=true,moveFocus=false) {
    if(!families.has(id))return;
    if(push&&!restoring)history.replaceState({...history.state,topicFocus:id},"",location.href);
    activeFamily=id;selected=null;$("search").value="";
    $("detail").innerHTML='<h2>Choose a concept</h2><p>Select an idea from the concept list to find its explanation and connections.</p>';
    $("breadcrumb").textContent=families.get(id).label;$("tabs").hidden=true;$("connections").innerHTML="";$("map-panel").hidden=true;
    renderNavigation();setStep("concepts");save(push);
    if(!restoring&&(moveFocus||matchMedia("(max-width:700px)").matches)){$("concepts").focus({preventScroll:true});$("concepts").scrollIntoView({block:"start"});}
  }
  function setTab(tab,push=true) {
    activeTab=tab;const map=tab==="map";
    $("text-tab").setAttribute("aria-selected",String(!map));$("map-tab").setAttribute("aria-selected",String(map));
    $("text-tab").tabIndex=map?-1:0;$("map-tab").tabIndex=map?0:-1;
    $("connections").hidden=map;$("map-panel").hidden=!map;
    if(map&&selected){renderMap();}
    if(push!==null)save(push);
  }
  function showDetails() {
    const n=nodes.get(selected), family=families.get(n.family),links=matchingLinks(selected);
    const edges=model.relationships.filter(e=>e.source===selected||e.target===selected);
    $("breadcrumb").innerHTML=`<span>Topics</span><span aria-hidden="true">/</span><span>${escape(family.label)}</span>`;
    const hierarchy=model.taxonomy?.hierarchy.filter(e=>e.narrower===selected)||[];
    const groupText=hierarchy.map(e=>model.taxonomy.topics.find(t=>t.id===e.broader)?.label).filter(Boolean);
    $("detail").innerHTML=`<h2 id="field-concept-title">${escape(n.label)}</h2><p class="field-definition">${escape(n.definition)}</p>${groupText.length?`<p class="field-parent">Part of the concept group: ${groupText.map(escape).join(", ")}</p>`:""}<div class="field-reading-header"><h3>Read in MLSysBook</h3><label for="field-book"${new Set(model.book_links.filter(l=>l.concept===selected).map(l=>l.volume)).size<=1&&!bookFilter?" hidden":""}>Book <select id="field-book"><option value="">All volumes</option><option value="vol1">Volume I</option><option value="vol2">Volume II</option><option value="vol3">Volume III</option><option value="vol4">Volume IV</option></select></label></div>${links.length?`<ul class="field-book-links">${links.map(l=>`<li><a href="${escape(l.url)}">${escape(l.label)}<span>${escape(l.volume.replace("vol","Volume "))} · ${escape(l.chapter_title)} →</span></a></li>`).join("")}</ul>`:`<p class="field-empty">${bookFilter?'No section is linked in this volume. Select “All volumes” to check the others.':"A book section has not been linked here yet."}</p>`}`;
    $("connections").innerHTML=`<div class="field-word-connections"><h3>How this idea connects</h3><ul>${edges.map(e=>`<li>${e.source===selected?escape(n.label):`<button type="button" data-concept="${escape(e.source)}">${escape(nodes.get(e.source).label)}</button>`} <strong>${escape(e.predicate)}</strong> ${e.target===selected?escape(n.label):`<button type="button" data-concept="${escape(e.target)}">${escape(nodes.get(e.target).label)}</button>`}<details><summary>When this applies</summary><p>${escape(e.conditions)}</p></details></li>`).join("")}</ul></div>`;
    $("topic-back").textContent=`← ${family.label}`;
    $("book").value=bookFilter;$("book").onchange=()=>{bookFilter=$("book").value;showDetails();if(svg)updateOverlay();($("book").parentElement.hidden?$("detail"):$("book")).focus({preventScroll:true});save();};
  }
  function navigate(id,moveFocus=false,push=true,keepTab=false) {
    if(!nodes.has(id))return;
    if(push&&!restoring)history.replaceState({...history.state,listFocus:id},"",location.href);
    if(svg&&viewName!=="focus")landscapeContext={view:viewName,box:box()};
    if(!keepTab)activeTab="text";
    selected=id;activeFamily=nodes.get(id).family;$("search").value="";
    showDetails();renderNavigation();$("tabs").hidden=false;
    setStep("detail");setTab(activeTab,null);
    if(svg)highlight();
    if(!restoring&&(moveFocus||matchMedia("(max-width:700px)").matches)){$("detail").focus({preventScroll:true});(matchMedia("(max-width:700px)").matches?document.querySelector(".field-reader"):$("detail")).scrollIntoView({block:"start"});}
    save(push);
  }
  document.querySelector(".field-explorer").addEventListener("click",e=>{
    const topic=e.target.closest("button[data-topic]");if(topic){chooseTopic(topic.dataset.topic,true,e.detail===0);return;}
    const b=e.target.closest("button[data-concept]");if(b)navigate(b.dataset.concept,e.detail===0);
  });
  $("topic-select").onchange=()=>chooseTopic($("topic-select").value);
  $("search").oninput=()=>{const push=!history.state?.q&&!!$("search").value;renderNavigation();setStep("concepts");save(push);};
  $("clear-search").onclick=()=>{$("search").value="";renderNavigation();$("search").focus();save();};
  function backToStep(step) {
    if(history.state?.from===step){history.back();return;}
    setStep(step);save(false);
    const target=step==="topics"?$("taxonomy").querySelector(`[data-topic="${CSS.escape(activeFamily)}"]`):$("list").querySelector(`[data-concept="${CSS.escape(selected||"")}"]`);
    (target||$(step==="topics"?"taxonomy":"concepts")).focus();
  }
  $("all-topics").onclick=()=>backToStep("topics");
  $("topic-back").onclick=()=>backToStep("concepts");
  $("text-tab").onclick=()=>setTab("text",false);$("map-tab").onclick=()=>{setTab("map",false);$("tabs").scrollIntoView({block:"start"});};
  $("tabs").onkeydown=e=>{if(["ArrowLeft","ArrowRight","Home","End"].includes(e.key)){e.preventDefault();const map=e.key==="End"||e.key==="ArrowRight";setTab(map?"map":"text",false);$(map?"map-tab":"text-tab").focus();}};
  $("view").onchange=()=>{landscapeContext=null;renderMap($("view").value);save();};
  $("fit").onclick=()=>{fit(visibleIds());save();};
  $("plus").onclick=()=>{zoom(.8);save();};$("minus").onclick=()=>{zoom(1.25);save();};
  $("neighbors").onclick=()=>{if(viewName!=="focus")landscapeContext={view:viewName,box:box()};renderMap();save();};
  $("return").onclick=()=>{if(!landscapeContext)return;const last=landscapeContext;landscapeContext=null;renderMap(last.view);setBox(last.box);save();};
  canvas.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || e.target.closest(".node")) return;
    drag = {x:e.clientX,y:e.clientY,view:box()}; canvas.setPointerCapture(e.pointerId); canvas.classList.add("dragging"); e.preventDefault();
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const [x,y,w,h] = drag.view, scale = Math.max(w/canvas.clientWidth,h/canvas.clientHeight);
    setBox([x-(e.clientX-drag.x)*scale,y-(e.clientY-drag.y)*scale,w,h]);
  });
  const endDrag = () => { if(drag)save();drag=null; canvas.classList.remove("dragging"); };
  canvas.addEventListener("pointerup",endDrag); canvas.addEventListener("pointercancel",endDrag); canvas.addEventListener("lostpointercapture",endDrag);
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault(); if (e.ctrlKey || e.metaKey) { zoom(Math.exp(Math.max(-.2,Math.min(.2,e.deltaY*.004))));save();return; }
    const [x,y,w,h] = box(), scale = Math.max(w/canvas.clientWidth,h/canvas.clientHeight);
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? canvas.clientHeight : 1;
    setBox([x+e.deltaX*unit*scale,y+e.deltaY*unit*scale,w,h]);save();
  }, {passive:false});
  canvas.addEventListener("keydown", (e) => {
    if (e.target !== canvas) return;
    if(e.key==="Escape"){e.preventDefault();setTab("text",false);$("text-tab").focus();return;}
    const [x,y,w,h] = box(), delta={ArrowLeft:[-.08*w,0],ArrowRight:[.08*w,0],ArrowUp:[0,-.08*h],ArrowDown:[0,.08*h]}[e.key];
    if (delta) { e.preventDefault(); setBox([x+delta[0],y+delta[1],w,h]);save(); }
    if (e.key === "+" || e.key === "=") { e.preventDefault(); zoom(.8); }
    if (e.key === "-") { e.preventDefault(); zoom(1.25); }
    if (e.key === "Home") { e.preventDefault(); $("fit").click(); }
  });
  function restore(event) {
    const previous=selected;
    const p=new URLSearchParams(location.hash.slice(1));restoring=true;
    bookFilter=["vol1","vol2","vol3","vol4"].includes(p.get("book"))?p.get("book"):"";
    activeTab=p.get("tab")==="map"?"map":"text";
    activeFamily=families.has(p.get("topic"))?p.get("topic"):"";
    if(nodes.has(p.get("concept")))navigate(p.get("concept"),false,false,true);
    else if(activeFamily)chooseTopic(activeFamily,false);
    else {selected=null;$("tabs").hidden=true;$("map-panel").hidden=true;$("connections").innerHTML="";$("detail").innerHTML='<h2>Explore an idea</h2><p>Choose a topic, then a concept. Its explanation, book sections, and connections will appear here.</p>';renderNavigation();setStep("topics");}
    $("search").value=p.get("q")||"";renderNavigation();
    if(p.get("q"))setStep("concepts");else if(["topics","concepts","detail"].includes(p.get("step")))setStep(p.get("step"));
    if(selected&&activeTab==="map"&&["overview","all"].includes(p.get("view")))renderMap(p.get("view"));
    if(svg&&history.state?.viewBox&&activeTab==="map")setBox(history.state.viewBox);
    restoring=false;save();
    if(event&&matchMedia("(max-width:700px)").matches){
      let target;
      if(mobileStep==="concepts")target=$("list").querySelector(`[data-concept="${CSS.escape(history.state?.listFocus||previous||"")}"]`);
      else if(mobileStep==="topics")target=$("taxonomy").querySelector(`[data-topic="${CSS.escape(history.state?.topicFocus||activeFamily)}"]`);
      else target=$("topic-back");
      if(target)requestAnimationFrame(()=>{target.focus({preventScroll:true});target.scrollIntoView({block:"center"});});
    }
  }
  document.querySelectorAll(".field-skip").forEach(link=>link.onclick=e=>{e.preventDefault();const target=document.querySelector(link.getAttribute("href"));target.focus({preventScroll:true});target.scrollIntoView({block:"start"});});
  const root=document.querySelector(".field-explorer"), header=document.querySelector("#quarto-header");
  const measureHeader=()=>root.style.setProperty("--field-top",`${Math.max(0,header?.getBoundingClientRect().bottom||0)+12}px`);
  if(header)new ResizeObserver(measureHeader).observe(header);
  window.addEventListener("resize",measureHeader);measureHeader();
  renderTopics();restore();window.addEventListener("popstate",restore);
})();
