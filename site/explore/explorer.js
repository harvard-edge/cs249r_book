(async () => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const fieldEntry = document.querySelector(".field-entry");
  if (fieldEntry) fieldEntry.href = window.parent !== window
    ? "https://mlsysbook.ai/explore/field.html"
    : location.pathname.endsWith("/atlas.html") ? "field-preview.html" : "field.html";
  if (window.parent !== window) document.body.classList.add("embedded-map");
  const esc = (value) =>
    String(value).replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  const initialDetail =
    "<h2>Read the relationships</h2><p>Select a concept to see its explanation, connected statements, and where it is taught.</p>";
  const bookNames = {
    vol1: "Volume I · Introduction",
    vol2: "Volume II · Scaling",
    vol3: "Volume III · Agentic systems",
    vol4: "Volume IV · Physical AI",
  };
  let catalog,
    data,
    graph,
    byId,
    selected = null,
    overviewReturn = null,
    loadVersion = 0,
    zoom = 1;
  let embedded;
  try {
    embedded = $("concept-data")
      ? JSON.parse($("concept-data").textContent)
      : null;
    if (embedded) catalog = embedded.catalog;
    else {
      const response = await fetch("catalog.json");
      if (!response.ok) throw Error("Catalog unavailable");
      catalog = await response.json();
    }
  } catch {
    $("concept-map").textContent =
      "The concept catalog could not load. Please reload the page.";
    return;
  }
  $("map-selector").innerHTML = catalog.maps
    .map((m) => `<option value="${esc(m.id)}">${esc(m.title)}</option>`)
    .join("");
  function nodeElement(id) {
    return $("concept-map").querySelector(`[id="concept-${CSS.escape(id)}"]`);
  }
  function edgeElement(id) {
    return $("concept-map").querySelector(`[id="relation-${CSS.escape(id)}"]`);
  }
  function emphasize(ids) {
    const activeNodes = new Set(ids),
      activeEdges = new Set();
    for (const id of ids) {
      const edges = graph
        ? graph
            .getElementById(id)
            .connectedEdges()
            .map((e) => e.data())
        : data.edges.filter((e) => e.source === id || e.target === id);
      edges.forEach((e) => {
        activeNodes.add(e.source);
        activeNodes.add(e.target);
        activeEdges.add(e.id);
      });
    }
    data.nodes.forEach((n) => {
      const el = nodeElement(n.id);
      el.classList.toggle("muted", ids.length > 0 && !activeNodes.has(n.id));
      el.classList.toggle("selected", n.id === selected);
      el.setAttribute("aria-pressed", String(n.id === selected));
    });
    data.edges.forEach((e) => {
      const el = edgeElement(e.id);
      el.classList.toggle("muted", ids.length > 0 && !activeEdges.has(e.id));
      el.classList.toggle("active", activeEdges.has(e.id));
    });
  }
  function setZoom(value) {
    const svg = $("concept-map").querySelector("svg");
    if (!svg) return;
    zoom = Math.min(8, Math.max(0.5, value));
    const frame = document.querySelector(".map-frame");
    const available = Math.max(200, frame.clientWidth - 22);
    svg.style.width = `${Math.round(available * zoom)}px`;
    svg.style.minWidth = "0";
    svg.style.maxWidth = "none";
    $("zoom-status").textContent = `${Math.round(zoom * 100)}%`;
    $("zoom-out").disabled = zoom <= 0.5;
    $("zoom-in").disabled = zoom >= 8;
  }
  function saveSelection() {
    const url = new URL(location.href);
    if (url.protocol === "file:") {
      url.search = "";
      const fragment = new URLSearchParams({ map: data.id });
      if (selected) fragment.set("concept", selected);
      if (atlas?.chapter()) fragment.set("chapter", atlas.chapter());
      url.hash = fragment.toString();
    } else {
      url.searchParams.set("map", data.id);
      url.hash = selected ? `concept=${encodeURIComponent(selected)}` : "";
      if (atlas?.chapter()) url.searchParams.set("chapter", atlas.chapter());
      else url.searchParams.delete("chapter");
    }
    history.replaceState(null, "", url);
    $("expand-map").href = url.href;
    if (window.parent !== window) {
      window.parent.postMessage(
        {
          type: "mlsys-concept-map-state",
          map: data.id,
          concept: selected,
          chapter: atlas?.chapter() || "",
        },
        location.origin === "null" ? "*" : location.origin,
      );
    }
  }
  function evidenceHTML(edge) {
    if (!Array.isArray(edge.evidence))
      return `<blockquote>${esc(edge.evidence?.quote || "")}</blockquote><a href="${esc(edge.evidence_url)}" target="_top">Read the supporting section ↗</a>`;
    const volume = edge.chapter.split("/")[0];
    return edge.evidence
      .map((item) => {
        const owner = catalog.volumes
          .flatMap((v) => v.chapters)
          .find((ch) => ch.id === volume + "/" + item.chapter);
        return `<p><strong>${esc(item.concept)}</strong> — ${esc(owner.title)}</p><p>${esc(item.quote)}</p><a href="${esc(owner.url)}" target="_top">Read the supporting chapter ↗</a>`;
      })
      .join("");
  }
  function detailMapsHTML(node) {
    const views = (node.detail_maps || []).filter((m) => catalog.maps.some((v) => v.id === m.id));
    return views.length ? `<h3>Unpack this idea</h3><p>Open a focused question with finer concepts and section-level evidence.</p>${views.map((m) => `<button type="button" data-detail-map="${esc(m.id)}">${esc(m.title)}</button>`).join("")}` : "";
  }
  function resourcesHTML(node) {
    if (!node.resources?.length) return "";
    return `<h3>Explore in practice</h3><ul>${node.resources.map((r) => `<li><a class="read-link" href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.title)} ↗</a><small>${esc(r.kind)} · ${r.status === "skeleton" ? "Planned activity — source outline only" : esc(r.status)} · repository source</small></li>`).join("")}</ul>`;
  }
  function select(id, focusDetail = false) {
    if (!byId.has(id)) return;
    selected = id;
    const n = byId.get(id),
      relations = data.edges.filter((e) => e.source === id || e.target === id);
    $("concept-detail").innerHTML =
      `<h2>${esc(n.label)}</h2><p>${esc(n.summary)}</p><p class="detail-source">Volume ${n.volume} · ${esc(catalog.volumes.flatMap((v) => v.chapters).find((ch) => ch.id === n.chapter)?.title || n.concept)}</p><a class="read-link" href="${esc(n.url)}" target="_top">Read the explanation in the book ↗</a>${detailMapsHTML(n)}${resourcesHTML(n)}<h3>Connected statements</h3><ul class="statements">${relations.map((e) => `<li><button data-concept="${esc(e.source)}">${esc(byId.get(e.source).label)}</button> <strong>${esc(e.label.replaceAll("\n", " "))}</strong> <button data-concept="${esc(e.target)}">${esc(byId.get(e.target).label)}</button><small>${esc(e.explanation)}</small><details class="relationship-evidence"><summary>Conditions and supporting explanation</summary><p>${esc(e.conditions)}</p>${evidenceHTML(e)}</details></li>`).join("")}</ul>`;
    $("concept-detail")
      .querySelectorAll("[data-concept]")
      .forEach((b) => (b.onclick = () => select(b.dataset.concept, true)));
    $("concept-detail").querySelectorAll("[data-detail-map]").forEach((b) => (b.onclick = () => { if (data.scope === "volume") overviewReturn = { map: data.id, concept: id }; loadMap(b.dataset.detailMap); }));
    emphasize([id]);
    renderIndex();
    $("focus-connections").disabled = false;
    saveSelection();
    if (focusDetail) $("concept-detail").focus({ preventScroll: false });
  }
  function matches() {
    const q = $("concept-search").value.trim().toLowerCase();
    return data.nodes.filter((n) =>
      `${n.label} ${(n.aliases || []).join(" ")} ${n.concept} ${n.summary}`.toLowerCase().includes(q),
    );
  }
  function renderIndex() {
    const nodes = matches();
    $("result-status").textContent =
      `${nodes.length} ${nodes.length === 1 ? "concept" : "concepts"}${$("concept-search").value ? " matching your search" : ""}`;
    $("concept-list").innerHTML = nodes
      .map(
        (n) =>
          `<button data-concept="${esc(n.id)}" aria-pressed="${n.id === selected}">${esc(n.label)}</button>`,
      )
      .join("");
    $("concept-list")
      .querySelectorAll("button")
      .forEach((b) => (b.onclick = () => select(b.dataset.concept, true)));
  }
  function renderCoverage() {
    const coverage = data.coverage;
    const box = $("map-coverage");
    if (!box) return;
    if (data.scope !== "volume" || !coverage) { box.replaceChildren(); return; }
    box.innerHTML = `<details><summary>What does this map cover?</summary><p>This view represents ${coverage.represented_goals} of ${coverage.teaching_goals} chapter teaching goals across ${coverage.represented_chapters} of ${coverage.teaching_chapters} teaching chapters. A teaching goal may contain several concepts. These counts describe the selected view, not completeness of the field.</p>${coverage.chapters.map((c) => `<details><summary>${esc(c.title)} · ${c.represented.length} goals represented</summary>${c.primary_outside_view.length ? `<p>Primary teaching goals outside this view:</p><ul>${c.primary_outside_view.map((g) => `<li>${esc(g)}</li>`).join("")}</ul>` : "<p>Each primary teaching goal has a concept in this view.</p>"}</details>`).join("")}</details>`;
  }
  function renderChecks() {
    $("learning-checks").replaceChildren();
    (data.checks || []).forEach((check, index) => {
      const box = document.createElement("section");
      box.className = "learning-check";
      box.innerHTML = `<h3>Check understanding</h3><p>${esc(check.question)}</p><fieldset><legend class="visually-hidden">Choose an answer</legend>${check.choices.map((choice, i) => `<label><input type="radio" name="check-${index}" value="${i}"><span>${esc(choice)}</span></label>`).join("")}</fieldset><button type="button">Check my reasoning</button><p class="check-feedback" role="status"></p>`;
      box.querySelector("button").onclick = () => {
        const answer = box.querySelector("input:checked"),
          feedback = box.querySelector(".check-feedback");
        if (!answer) {
          feedback.textContent = "Choose an answer first.";
          return;
        }
        feedback.textContent =
          (Number(answer.value) === check.answer
            ? "Yes. "
            : "Reconsider the relationship. ") + check.explanation;
      };
      $("learning-checks").append(box);
    });
  }
  let atlas;
  // Preserve node clicks and native touch scrolling; drag the background on desktop.
  const panFrame = document.querySelector(".map-frame");
  let panDrag = null;
  panFrame.tabIndex = 0;
  panFrame.setAttribute("aria-label", "Concept map viewport. Drag the background or use arrow keys to pan.");
  panFrame.addEventListener("pointerdown", (event) => {
    if (event.pointerType !== "mouse" || event.button !== 0 || event.target.closest(".node")) return;
    panDrag = {x:event.clientX,y:event.clientY,left:panFrame.scrollLeft,top:panFrame.scrollTop};
    panFrame.setPointerCapture(event.pointerId);
    panFrame.classList.add("dragging");
    event.preventDefault();
  });
  panFrame.addEventListener("pointermove", (event) => {
    if (!panDrag) return;
    panFrame.scrollLeft = panDrag.left - event.clientX + panDrag.x;
    panFrame.scrollTop = panDrag.top - event.clientY + panDrag.y;
  });
  const endPan = () => { panDrag = null; panFrame.classList.remove("dragging"); };
  panFrame.addEventListener("pointerup", endPan);
  panFrame.addEventListener("pointercancel", endPan);
  panFrame.addEventListener("lostpointercapture", endPan);
  function focusNodes(ids) {
    const elements = ids.map(nodeElement).filter(Boolean);
    if (!elements.length) return;
    const bounds = elements.map((el) => el.getBBox());
    const left = Math.min(...bounds.map((b) => b.x)),
      top = Math.min(...bounds.map((b) => b.y));
    const right = Math.max(...bounds.map((b) => b.x + b.width)),
      bottom = Math.max(...bounds.map((b) => b.y + b.height));
    const svg = $("concept-map").querySelector("svg"),
      frame = document.querySelector(".map-frame");
    const view = svg.viewBox.baseVal;
    const target = Math.min(
      5,
      Math.max(1.5, view.width / Math.max(300, right - left + 180)),
    );
    setZoom(target);
    const frameBox = frame.getBoundingClientRect();
    const matrix = elements[0].getScreenCTM();
    const point = svg.createSVGPoint();
    point.x = (left + right) / 2;
    point.y = (top + bottom) / 2;
    const screen = point.matrixTransform(matrix);
    frame.scrollTo({
      left: frame.scrollLeft + screen.x - frameBox.left - frame.clientWidth / 2,
      top: frame.scrollTop + screen.y - frameBox.top - frame.clientHeight / 2,
      behavior: "auto",
    });
  }
  function returnDestination() {
    if (!data || data.scope === "volume") return null;
    const volume = data.chapter.split("/")[0];
    if (overviewReturn?.map.startsWith(volume + ".")) return overviewReturn;
    const primary = catalog.maps.find((m) => m.scope === "volume" && m.chapter.startsWith(volume + "/"));
    return primary ? { map: primary.id, concept: null } : null;
  }
  function navigateMap(id) {
    overviewReturn = null;
    return loadMap(id);
  }
  async function loadMap(id, concept = null) {
    if (!id) return;

    const version = ++loadVersion;
    const directory = `maps/${encodeURIComponent(id)}`;
    $("concept-map").textContent = "Loading the concept map…";
    try {
      let nextData, svgText;
      if (embedded && embedded.maps[id]) {
        nextData = embedded.maps[id].graph;
        svgText = embedded.maps[id].svg;
      } else {
        const responses = await Promise.all([
          fetch(`${directory}/graph.json`),
          fetch(`${directory}/map.svg`),
        ]);
        if (responses.some((r) => !r.ok)) throw Error("Map unavailable");
        nextData = await responses[0].json();
        svgText = await responses[1].text();
      }
      const doc = new DOMParser().parseFromString(svgText, "image/svg+xml");
      if (doc.querySelector("parsererror")) throw Error("Invalid SVG");
      if (version !== loadVersion) return;
      if (graph) graph.destroy();
      data = nextData;
      byId = new Map(data.nodes.map((n) => [n.id, n]));
      selected = null;
      $("concept-map").replaceChildren(
        document.importNode(doc.documentElement, true),
      );
      graph =
        typeof cytoscape === "undefined"
          ? null
          : cytoscape({
              headless: true,
              styleEnabled: false,
              elements: [
                ...data.nodes.map((n) => ({ data: n })),
                ...data.edges.map((e) => ({ data: e })),
              ],
            });
      $("return-overview").hidden = !returnDestination();
      $("map-title").textContent = data.title;
      $("map-description").textContent = data.description;
      $("focus-question").textContent = data.title;
      const chapter = catalog.volumes
        .flatMap((v) => v.chapters)
        .find((ch) => ch.id === data.chapter);
      $("map-context").textContent =
        `${bookNames[data.chapter.split("/")[0]]} · ${data.scope === "volume" ? "Volume overview" : chapter.title} · ${data.edges.some((e) => e.status !== "reviewed") ? "Draft relationship map" : "Concept map"}`;
      $("download-map").href = `${directory}/map.svg`;
      $("download-map").download = `${id}.svg`;
      $("map-selector").value = id;
      $("concept-search").value = "";
      $("concept-detail").innerHTML = initialDetail;
      data.nodes.forEach((n) => {
        const el = nodeElement(n.id);
        el.setAttribute("tabindex", "0");
        el.setAttribute("role", "button");
        el.setAttribute("aria-label", n.label);
        el.onclick = () => select(n.id);
        el.onkeydown = (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            select(n.id);
          }
        };
      });
      zoom = 1;
      setZoom(zoom);
      renderIndex();
      renderChecks();
      renderCoverage();
      $("focus-connections").disabled = true;
      atlas?.mapLoaded(data);
      if (byId.has(concept)) select(concept);
      else {
        emphasize([]);
        saveSelection();
      }
    } catch {
      if (version === loadVersion)
        $("concept-map").textContent =
          "This map could not load. Choose another question or reload the page.";
    }
  }
  $("return-overview").onclick = () => { const back = returnDestination(); if (back) loadMap(back.map, back.concept); };
  $("zoom-in").onclick = () => setZoom(zoom + 0.25);
  $("zoom-out").onclick = () => setZoom(zoom - 0.25);
  $("zoom-fit").onclick = () => {
    setZoom(1);
    document.querySelector(".map-frame").scrollTo({ left: 0, top: 0 });
  };
  $("expand-map").hidden = window.parent === window;
  new ResizeObserver(() => setZoom(zoom)).observe(
    document.querySelector(".map-frame"),
  );
  $("map-selector").onchange = () => navigateMap($("map-selector").value);
  $("concept-search").oninput = () => {
    if (!data) return;
    selected = null;
    renderIndex();
    const ids = matches().map((n) => n.id);
    emphasize(
      $("concept-search").value ? (ids.length ? ids : ["no-match"]) : [],
    );
  };
  $("reset-view").onclick = () => {
    if (!data) return;
    selected = null;
    $("concept-search").value = "";
    emphasize([]);
    renderIndex();
    saveSelection();
    $("concept-detail").innerHTML = initialDetail;
    $("focus-connections").disabled = true;
  };
  window.addEventListener("hashchange", () => {
    if (data) {
      const id = new URLSearchParams(location.hash.slice(1)).get("concept");
      if (byId.has(id)) select(id);
    }
  });
  atlas = window.MLSysVolume.init(catalog, navigateMap, focusNodes, saveSelection);
  $("focus-connections").onclick = () => {
    if (!selected) return;
    focusNodes([
      selected,
      ...data.edges
        .filter((e) => e.source === selected || e.target === selected)
        .flatMap((e) => [e.source, e.target]),
    ]);
  };
  if (!catalog.maps.length) {
    $("concept-map").textContent =
      "No concept maps have been authored for this scope yet.";
    return;
  }
  const requested =
    new URLSearchParams(location.search).get("map") ||
    new URLSearchParams(location.hash.slice(1)).get("map");
  const initial = catalog.maps.some((m) => m.id === requested)
    ? requested
    : (
        catalog.maps.find(
          (m) =>
            m.scope === "volume" &&
            (!catalog.chapter ||
              m.chapter.split("/")[0] === catalog.chapter.split("/")[0]),
        ) || catalog.maps[0]
      ).id;
  await loadMap(
    initial,
    new URLSearchParams(location.hash.slice(1)).get("concept"),
  );
})();
