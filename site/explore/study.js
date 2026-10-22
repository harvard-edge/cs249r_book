// ML Systems Taxonomy, presented the way established taxonomies are (MITRE
// ATT&CK/ATLAS, SFIA, ESCO): the whole taxonomy as one matrix, and an entry
// page per area, subject, and concept with a sidebar tree, a facts box, and
// tables. Concept pages carry a local concept map. Every page has a hash:
//   #                         matrix          #area=<id>          area page
//   #subject=<id>             subject page    #concept=<id>[&to=] concept page
//   #view=concept             concept map     #view=glossary[&term=<id>]
// Older hashes (#row=, #view=graph…, #view=map, #view=index) still resolve.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const push = (map, key, value) => (map.get(key) || map.set(key, []).get(key)).push(value);
  const cap = (text) => text.charAt(0).toUpperCase() + text.slice(1);
  const VOLUME = { vol1: 'Vol I', vol2: 'Vol II', vol3: 'Vol III', vol4: 'Vol IV' };
  const BAND_ORDER = ['lifecycle', 'stack', 'qualities', 'foundations', 'system-types'];
  // The kinds of system a reader can highlight in the matrix (context facets).
  const LENSES = [['generative', 'Generative'], ['agentic', 'Agentic'], ['physical', 'Physical'], ['edge', 'Edge'],
    ['distributed', 'Distributed'], ['recommendation', 'Recommendation'], ['multimodal', 'Multimodal']];
  // Incoming relations a practitioner pulls on: "what reduces / limits X?"
  const LEVERS = ['reduces', 'increases', 'limits', 'determines', 'mitigates', 'causes'];
  const MAP_LIMIT = 6;

  let data, groups, areas, rows, concepts, homeOf, areaOf, outgoing, incoming, secondaryOf, linksOf, gapsOf, relations, neighbors, lenses;
  const state = { view: 'home', area: null, subject: null, concept: null, to: null, term: null, lens: null, showConcepts: false };
  const visited = [];

  async function load() {
    try {
      const response = await fetch('framework-data.json', { cache: 'no-cache' });
      if (!response.ok) throw new Error(response.statusText);
      data = await response.json();
    } catch (error) {
      $('bok-matrix').innerHTML = '<p class="bok-loading">The taxonomy could not be loaded. Read the <a href="framework-reference.md">written taxonomy</a> instead.</p>';
      return;
    }
    index();
    renderMeta();
    renderLenses();
    renderMatrix();
    renderRelease();
    renderGlossary();
    renderPrintOutline();
    bind();
    restore();
  }

  function index() {
    const fw = data.framework;
    groups = fw.groups?.length ? fw.groups : [{ id: 'all', label: 'Areas', question: '' }];
    areas = fw.areas.map((a) => ({ ...a, group: a.group || 'all' }));
    rows = new Map();
    areaOf = new Map();
    homeOf = new Map();
    secondaryOf = new Map();
    for (const area of areas) {
      for (const row of area.rows) {
        rows.set(row.id, row);
        areaOf.set(row.id, area);
        for (const id of row.concepts) homeOf.set(id, row);
        for (const id of row.secondary_concepts || []) push(secondaryOf, id, row);
      }
    }
    concepts = new Map(data.concepts.map((c) => [c.id, c]));
    relations = new Map((data.relations || []).map((r) => [r.id, r]));
    outgoing = new Map();
    incoming = new Map();
    neighbors = new Map();
    for (const rel of data.relationships) {
      push(outgoing, rel.source, rel);
      push(incoming, rel.target, rel);
      push(neighbors, rel.source, rel.target);
      push(neighbors, rel.target, rel.source);
    }
    linksOf = new Map();
    for (const link of data.book_links) push(linksOf, link.concept, link);
    gapsOf = new Map();
    for (const gap of fw.gaps || []) if (gap.subject_row) push(gapsOf, gap.subject_row, gap);
    const used = new Set(areas.flatMap((a) => a.rows.flatMap((r) => r.contexts || [])));
    lenses = LENSES.filter(([id]) => used.has(id));
  }

  const groupOf = (area) => groups.find((g) => g.id === area.group);
  const conceptCount = (area) => area.rows.reduce((n, r) => n + r.concepts.length, 0);
  const code = (value) => (value ? `<span class="bok-code">${esc(value)}</span>` : '');
  const subjectHref = (row) => `#subject=${encodeURIComponent(row.id)}`;
  const conceptHref = (id) => `#concept=${encodeURIComponent(id)}`;
  const conceptLink = (id) => `<a href="${conceptHref(id)}"${concepts.get(id)?.scope === 'background' ? ' class="is-background"' : ''}>${esc(concepts.get(id)?.label || id)}</a>`;
  const subjectLink = (row) => `<a href="${subjectHref(row)}">${code(row.code)}${esc(row.label)}</a>`;
  const verb = (rel, direction) => {
    const r = relations.get(rel.relation);
    return r ? (direction === 'out' ? r.forward : r.inverse) : rel.predicate;
  };
  const bandOrder = () => [...BAND_ORDER.filter((id) => groups.some((g) => g.id === id)), ...groups.map((g) => g.id).filter((id) => !BAND_ORDER.includes(id))];

  function renderMeta() {
    const fw = data.framework;
    $('bok-stats').textContent = `${areas.length} areas · ${rows.size} subjects · ${concepts.size} concepts · ${data.relationships.length} relationships`;
    $('bok-version').textContent = fw.version ? `v${fw.version.replace(/-draft$/, '')}` : '';
    const gapList = (fw.gaps || []).map((g) => {
      const row = rows.get(g.subject_row);
      return `<li><b>${esc(g.topic || g.label)}</b>${row ? ` · ${subjectLink(row)}` : ''}</li>`;
    }).join('');
    if (gapList) $('bok-gaps').innerHTML = `<p><strong>Known gaps.</strong> Topics this version does not yet cover, declared so they are not mistaken for omissions:</p><ul>${gapList}</ul>`;
    const rules = (fw.tagging_rules || []).map((r) => `<li>${esc(r)}</li>`).join('');
    const home = (fw.rationale || []).find((r) => r.startsWith('Home rule.'));
    const cases = (fw.placement_examples || []).filter((x) => concepts.has(x.concept) && rows.has(x.home))
      .map((x) => `<li>${conceptLink(x.concept)} → ${subjectLink(rows.get(x.home))}: ${esc(x.why)}</li>`).join('');
    if (rules) $('bok-tagging').innerHTML = `<p><strong>How to tag with it.</strong> Chapters, quizzes, questions, slides, and labs are tagged with the same concepts and subjects:</p><ul class="bok-cases">${rules}</ul>`
      + (home ? `<p><strong>Where a concept lives.</strong> ${esc(home.replace(/^Home rule\. /, ''))}</p>` : '')
      + (cases ? `<ul class="bok-cases">${cases}</ul>` : '');
    $('bok-method').textContent = `Version ${fw.version}. Generated from shared/concepts/ml-systems-framework.yml and ml-systems-field.yml.`;
  }

  // -------------------------------------------------------------------- matrix
  // One band per group; each area is a column and each subject a cell.
  function renderLenses() {
    $('bok-lenses').innerHTML = '<span class="bok-filter-label">Highlight</span>' + lenses.map(([id, label]) =>
      `<button type="button" data-lens="${esc(id)}" aria-pressed="false">${esc(label)}</button>`).join('');
  }

  function renderMatrix() {
    const bands = bandOrder().map((id) => {
      const group = groups.find((g) => g.id === id);
      const members = areas.filter((a) => a.group === id);
      const columns = members.map((area) => `<div class="bok-col">
          <a class="bok-col-head" href="#area=${esc(area.id)}">${code(area.code)}<span>${esc(area.display_label || area.label)}</span></a>
          <p class="bok-col-count">${area.rows.length} subjects</p>
          <ul>${area.rows.map((row) => {
            const hit = !state.lens || (row.contexts || []).includes(state.lens);
            const list = state.showConcepts ? `<span class="bok-cell-concepts">${row.concepts.map((cid) => esc(concepts.get(cid)?.label)).join(' · ')}</span>` : '';
            return `<li><a class="bok-cell${hit ? '' : ' is-dim'}" href="${subjectHref(row)}" title="${esc(row.definition)}">${code(row.code)}<span class="bok-cell-label">${esc(row.display_label || row.label)}</span>${list}</a></li>`;
          }).join('')}</ul></div>`).join('');
      return `<section class="bok-band" aria-labelledby="band-${esc(id)}">
        <h3 class="bok-band-head" id="band-${esc(id)}">${code(group.code)}${esc(group.label)}<span>${esc(group.question)}</span></h3>
        <div class="bok-cols">${columns}</div></section>`;
    }).join('');
    $('bok-matrix').innerHTML = bands;
    for (const b of document.querySelectorAll('#bok-lenses button')) b.setAttribute('aria-pressed', String(b.dataset.lens === state.lens));
    $('bok-show-concepts').setAttribute('aria-pressed', String(state.showConcepts));
    $('bok-show-concepts').textContent = state.showConcepts ? 'Hide concepts' : 'Show concepts';
  }

  function renderRelease() {
    const fw = data.framework;
    const citation = `Vijay Janapa Reddi. ML Systems Taxonomy, version ${fw.version}. Machine Learning Systems, 2026. https://mlsysbook.ai/explore/`;
    $('bok-cite-text').textContent = citation;
    $('bok-cite-meta').innerHTML = `Version ${esc(fw.version)} · Licensed <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> · Editorial draft for expert review`;
    $('bok-copy-cite').addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(citation); $('bok-copy-cite').textContent = 'Copied'; } catch { $('bok-copy-cite').textContent = 'Select the text to copy'; }
      setTimeout(() => { $('bok-copy-cite').textContent = 'Copy citation'; }, 1800);
    });
  }

  // The full outline exists only for printing; on screen the matrix and the
  // entry pages carry the same content.
  function renderPrintOutline() {
    $('bok-print-outline').innerHTML = `<h2>The ML Systems Taxonomy, version ${esc(data.framework.version)}</h2>` + bandOrder().map((id) => {
      const group = groups.find((g) => g.id === id);
      return `<h3>${esc(group.code || '')} ${esc(group.label)}</h3>` + areas.filter((a) => a.group === id).map((area) =>
        `<h4>${esc(area.code || '')} ${esc(area.label)}</h4><dl>${area.rows.map((row) =>
          `<dt>${esc(row.code || '')} ${esc(row.label)}</dt><dd>${esc(row.definition)}<br><i>${row.concepts.map((cid) => esc(concepts.get(cid)?.label)).join(' · ')}</i></dd>`).join('')}</dl>`).join('');
    }).join('');
  }

  // --------------------------------------------------------------- entry pages
  // Sidebar tree: groups, areas, and the subjects of the current area.
  function renderTree(activeArea, activeSubject) {
    $('bok-tree').innerHTML = `<details class="bok-tree-box" open><summary>Browse the taxonomy</summary>` + bandOrder().map((id) => {
      const group = groups.find((g) => g.id === id);
      return `<p class="bok-tree-group">${esc(group.label)}</p><ul>${areas.filter((a) => a.group === id).map((area) => {
        const open = area.id === activeArea;
        const subjects = open ? `<ul>${area.rows.map((row) => `<li><a href="${subjectHref(row)}"${row.id === activeSubject ? ' aria-current="page"' : ''}>${code(row.code)}${esc(row.display_label || row.label)}</a></li>`).join('')}</ul>` : '';
        return `<li><a href="#area=${esc(area.id)}"${open && !activeSubject ? ' aria-current="page"' : ''}${open ? ' class="is-open"' : ''}>${code(area.code)}${esc(area.display_label || area.label)}</a>${subjects}</li>`;
      }).join('')}</ul>`;
    }).join('') + '</details>';
  }

  function crumbs(parts) {
    $('bok-crumbs').innerHTML = [`<a href="#">Matrix</a>`, ...parts].join('<span aria-hidden="true">›</span>');
  }

  const facts = (items) => `<div class="bok-facts" role="group" aria-label="Facts">${items.filter(([, v]) => v).map(([k, v]) => `<p><span class="bok-facts-label">${esc(k)}</span>${v}</p>`).join('')}</div>`;

  function scopeNote(text) {
    return esc(text).replace(/\[\[([^\]]+)\]\]/g, (_, id) => {
      const row = rows.get(id);
      if (row) return subjectLink(row);
      const area = areas.find((a) => a.id === id);
      return area ? `<a href="#area=${esc(area.id)}">${code(area.code)}${esc(area.label)}</a>` : esc(id);
    });
  }

  function conceptTable(ids, caption) {
    if (!ids.length) return '';
    return `<h3 class="bok-section">${esc(caption)}</h3><table class="bok-table"><thead><tr><th>Concept</th><th>Definition</th></tr></thead><tbody>${ids.map((id) =>
      `<tr><td>${conceptLink(id)}${concepts.get(id)?.scope === 'background' ? ' <span class="bok-tag">background</span>' : ''}</td><td>${esc(concepts.get(id)?.definition)}</td></tr>`).join('')}</tbody></table>`;
  }

  function renderArea() {
    const area = areas.find((a) => a.id === state.area);
    const group = groupOf(area);
    renderTree(area.id, null);
    crumbs([`<span>${esc(group.label)}</span>`, `<span aria-current="page">${esc(area.code || '')} ${esc(area.label)}</span>`]);
    $('bok-entry-body').innerHTML = `<div class="bok-entry-head">
        <div><h2>${code(area.code)}${esc(area.label)}</h2><p class="bok-lead">${esc(area.question)}</p><p class="bok-text">${esc(area.scope)}</p></div>
        ${facts([['Code', esc(area.code)], ['Group', esc(group.label)], ['Subjects', String(area.rows.length)], ['Concepts', String(conceptCount(area))]])}
      </div>
      <h3 class="bok-section">Subjects</h3><table class="bok-table"><thead><tr><th>Code</th><th>Subject</th><th>Definition</th><th class="bok-num">Concepts</th></tr></thead><tbody>${area.rows.map((row) =>
        `<tr><td class="bok-mono">${esc(row.code || '')}</td><td><a href="${subjectHref(row)}">${esc(row.label)}</a></td><td>${esc(row.definition)}</td><td class="bok-num">${row.concepts.length}</td></tr>`).join('')}</tbody></table>`;
  }

  function renderSubject() {
    const row = rows.get(state.subject);
    const area = areaOf.get(row.id);
    const group = groupOf(area);
    renderTree(area.id, row.id);
    crumbs([`<a href="#area=${esc(area.id)}">${esc(area.code || '')} ${esc(area.label)}</a>`, `<span aria-current="page">${esc(row.code || '')}</span>`]);
    const contexts = (row.contexts || []).map((id) => lenses.find(([l]) => l === id)?.[1]).filter(Boolean).join(', ');
    const chapters = (row.book_chapters || []).map((c) => `<a href="${esc(c.url)}">${esc(VOLUME[c.volume] || c.volume)}: ${esc(c.title)}</a>`).join('<br>');
    const elsewhere = (row.excludes || []).map((t) => `<li>${scopeNote(t)}</li>`).join('');
    const covers = (row.includes || []).map((t) => `<li>${scopeNote(t)}</li>`).join('');
    const gaps = (gapsOf.get(row.id) || []).map((g) => `<li><b>${esc(g.topic || g.label)}.</b> ${esc(g.reason.replace(/^Known gap for v[\d.]+\.\s*/, ''))}</li>`).join('');
    $('bok-entry-body').innerHTML = `<div class="bok-entry-head">
        <div><h2>${code(row.code)}${esc(row.label)}</h2><p class="bok-lead">${esc(row.definition)}</p>
          ${covers ? `<p class="bok-text-label">Covers</p><ul class="bok-list">${covers}</ul>` : ''}</div>
        ${facts([['Code', esc(row.code)], ['Area', `<a href="#area=${esc(area.id)}">${esc(area.code || '')} ${esc(area.label)}</a>`], ['Group', esc(group.label)],
          ['Concepts', String(row.concepts.length)], ['Systems', esc(contexts)], ['Taught in', chapters]])}
      </div>
      ${conceptTable(row.concepts, 'Concepts')}
      ${conceptTable(row.secondary_concepts || [], 'Related concepts from other subjects')}
      ${elsewhere ? `<h3 class="bok-section">Not in this subject</h3><ul class="bok-list">${elsewhere}</ul>` : ''}
      ${gaps ? `<h3 class="bok-section">Known gaps</h3><ul class="bok-list">${gaps}</ul>` : ''}`;
  }

  // ------------------------------------------------------------- concept map
  let cy = null;
  let palette = null;

  function readPalette() {
    const style = getComputedStyle($('bok-root'));
    const v = (name) => style.getPropertyValue(name).trim();
    palette = { ink: v('--bok-ink'), muted: v('--bok-muted'), faint: v('--bok-faint'), surface: v('--bok-surface'),
      bg: v('--bok-page') || v('--bok-surface'), line: v('--bok-line-strong'), edge: v('--bok-edge'), accent: v('--bok-accent') };
  }

  // Concepts are text chips: outlined ink, the centered concept solid,
  // background concepts dashed. Connectors share one spine per side.
  function mapStyle() {
    const c = palette;
    return [
      { selector: 'node', style: {
        shape: 'round-rectangle', width: 'data(w)', height: 30, 'background-color': c.surface, 'border-width': 1, 'border-color': c.line,
        label: 'data(label)', color: c.ink, 'font-family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif', 'font-size': 13,
        'font-weight': 500, 'text-valign': 'center', 'text-halign': 'center', 'text-wrap': 'ellipsis', 'text-max-width': 'data(tw)',
      } },
      { selector: 'node[?background]', style: { 'border-style': 'dashed', color: c.muted } },
      { selector: 'node.center', style: { height: 38, 'background-color': c.ink, 'border-color': c.ink, color: c.bg, 'font-size': 15, 'font-weight': 700 } },
      { selector: 'node.more', style: { 'border-width': 0, 'background-opacity': 0, color: c.faint, 'font-size': 12, events: 'no' } },
      { selector: 'edge', style: { width: 1.2, 'line-color': c.edge, 'target-arrow-color': c.edge, 'target-arrow-shape': 'triangle', 'arrow-scale': 0.7,
        'curve-style': 'taxi', 'taxi-direction': 'horizontal', 'taxi-turn': '50%' } },
      // Desktop: each line carries its verb above its own segment, next to the
      // outer concept, so every link reads left to right as a sentence.
      { selector: 'edge.into, edge.from', style: { 'font-size': 10.5, 'font-weight': 500, color: c.muted, 'font-family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif' } },
      { selector: 'edge.into', style: { 'taxi-turn': (e) => e.data('turn'), 'source-label': 'data(label)', 'source-text-offset': (e) => e.data('offset'), 'source-text-margin-y': -8 } },
      { selector: 'edge.from', style: { 'taxi-turn': (e) => e.data('turn'), 'target-label': 'data(label)', 'target-text-offset': (e) => e.data('offset'), 'target-text-margin-y': -8 } },
      { selector: 'node.hover', style: { 'border-color': c.accent, 'border-width': 2 } },
      { selector: 'edge.hover', style: { 'line-color': c.accent, 'target-arrow-color': c.accent, width: 2 } },
    ];
  }

  function ensureMap(container) {
    if (cy && cy.container() === container) return cy;
    if (typeof window.cytoscape !== 'function') return null;
    if (cy) cy.destroy();
    readPalette();
    cy = window.cytoscape({ container, style: mapStyle(), userZoomingEnabled: false, userPanningEnabled: false,
      boxSelectionEnabled: false, autoungrabify: true, autounselectify: true });
    cy.on('tap', 'node', (e) => { if (!e.target.hasClass('center') && concepts.has(e.target.id())) location.hash = conceptHref(e.target.id()); });
    cy.on('mouseover', 'node', (e) => {
      if (!concepts.has(e.target.id())) return;
      e.target.addClass('hover');
      e.target.connectedEdges().addClass('hover');
      container.style.cursor = e.target.hasClass('center') ? '' : 'pointer';
    });
    cy.on('mouseout', 'node', () => { cy.elements().removeClass('hover'); container.style.cursor = ''; });
    return cy;
  }

  const chipWidth = (label, max = 220) => Math.min(max, Math.round(label.length * 7.1 + 26));

  // Causes on the left, effects on the right, grouped by relation; at most
  // MAP_LIMIT per group, with every relationship in the table below. Text
  // stays at full size: the canvas grows to fit instead of zooming out.
  const COLUMN = 360;
  const SPINE = 150;
  function drawMap(id, container) {
    if (!ensureMap(container)) return;
    // Narrow screens get the relationships table instead of a drawing: a
    // map does not read at phone width, and the table carries the same links.
    const box = container.closest('.bok-mapbox');
    box.hidden = false;
    if (container.clientWidth < 640) { box.hidden = true; return; }
    const blocksFor = (list, direction) => {
      const byRelation = new Map();
      for (const rel of list) push(byRelation, rel.relation || rel.predicate, rel);
      return [...byRelation.entries()].sort((a, b) => (LEVERS.includes(b[0]) - LEVERS.includes(a[0])) || b[1].length - a[1].length)
        .map(([relation, rels]) => ({ title: relation, rels, direction }));
    };
    const els = [];
    const seen = new Set([id]);
    const centerWidth = Math.min(2 * SPINE - 16, Math.round(concepts.get(id).label.length * 8.8 + 40));
    const centerHalf = centerWidth / 2;
    const place = (blocks, x, top) => {
      let y = top;
      for (const block of blocks) {
        for (const rel of block.rels.slice(0, MAP_LIMIT)) {
          const other = block.direction === 'in' ? rel.source : rel.target;
          if (seen.has(other)) continue;
          seen.add(other);
          const c = concepts.get(other);
          const w = chipWidth(c.label, 200);
          els.push({ group: 'nodes', position: { x, y }, data: { id: other, label: c.label, w, tw: 180, background: c.scope === 'background' } });
          const into = rel.target === id;
          // The outer segment runs from the concept to the spine; the verb sits on it.
          const segment = COLUMN - w / 2 - SPINE;
          els.push({ group: 'edges', classes: into ? 'into' : 'from', data: { id: rel.id, source: rel.source, target: rel.target,
            label: relations.get(rel.relation)?.forward || rel.predicate, offset: segment / 2, turn: into ? segment : SPINE - centerHalf } });
          y += 44;
        }
        if (block.rels.length > MAP_LIMIT) {
          els.push({ group: 'nodes', classes: 'more', position: { x, y: y - 8 }, data: { id: `m-${block.direction}-${block.title}`, label: `+ ${block.rels.length - MAP_LIMIT} more in the table`, w: 160, tw: 160 } });
          y += 26;
        }
        y += 14;
      }
      return y - top;
    };
    const ins = blocksFor(incoming.get(id) || [], 'in');
    const outs = blocksFor(outgoing.get(id) || [], 'out');
    const center = concepts.get(id);
    const tall = Math.max(place(ins, -COLUMN, 30), place(outs, COLUMN, 30), 80);
    els.unshift({ group: 'nodes', classes: 'center', position: { x: 0, y: 30 + tall / 2 - 20 }, data: { id, label: center.label, w: centerWidth, tw: centerWidth - 20 } });
    const height = tall + 40;
    cy.elements().remove();
    cy.add(els);
    cy.layout({ name: 'preset', fit: false }).run();
    const zoom = Math.min(1, (container.clientWidth - 24) / cy.elements().boundingBox().w);
    container.style.height = `${Math.max(160, Math.round(height * zoom))}px`;
    cy.resize();
    cy.zoom(zoom);
    cy.center();
  }

  // Shortest chains of relationships between two concepts, read as sentences.
  function paths(from, to, k = 3) {
    const found = [];
    const banned = new Set();
    for (let attempt = 0; attempt < 12 && found.length < k; attempt++) {
      const prev = new Map([[from, null]]);
      const queue = [from];
      for (let i = 0; i < queue.length && !prev.has(to); i++) {
        for (const rel of [...(outgoing.get(queue[i]) || []), ...(incoming.get(queue[i]) || [])]) {
          if (banned.has(rel.id)) continue;
          const next = rel.source === queue[i] ? rel.target : rel.source;
          if (!prev.has(next)) { prev.set(next, { node: queue[i], rel }); queue.push(next); }
        }
      }
      if (!prev.has(to)) break;
      const steps = [];
      for (let x = to; prev.get(x); x = prev.get(x).node) steps.unshift({ from: prev.get(x).node, to: x, rel: prev.get(x).rel });
      const key = steps.map((s) => s.rel.id).join(',');
      if (!found.some((p) => p.key === key)) found.push({ key, steps });
      // Ban the busiest step so the next search finds a different route.
      banned.add(steps.reduce((best, s) => ((neighbors.get(s.to)?.length || 0) > (neighbors.get(best.to)?.length || 0) ? s : best), steps[0]).rel.id);
    }
    return found.sort((a, b) => a.steps.length - b.steps.length);
  }

  function relationshipTable(id) {
    const groupsIn = [];
    const groupsOut = [];
    for (const [direction, list, bucket] of [['in', incoming.get(id) || [], groupsIn], ['out', outgoing.get(id) || [], groupsOut]]) {
      const byRelation = new Map();
      for (const rel of list) push(byRelation, rel.relation || rel.predicate, rel);
      for (const [relation, rels] of byRelation) bucket.push({ relation, rels, direction, lever: direction === 'in' && LEVERS.includes(relation) });
    }
    const ordered = [...groupsIn.filter((g) => g.lever), ...groupsIn.filter((g) => !g.lever), ...groupsOut];
    if (!ordered.length) return '';
    // Each row is a complete proposition ("A increases B"), so it reads
    // correctly on its own, on any screen, without headers or arrows.
    const self = `<b>${esc(concepts.get(id).label)}</b>`;
    const name = (other) => (other === id ? self : conceptLink(other));
    return `<h3 class="bok-section">Relationships</h3><table class="bok-table bok-rels"><thead><tr><th>Relationship</th><th>Holds when</th></tr></thead><tbody>${ordered.map((g) =>
      g.rels.map((rel, i) => `<tr${i === 0 ? ' class="bok-group-start"' : ''}><td>${name(rel.source)} <span class="bok-verb">${esc(relations.get(rel.relation)?.forward || rel.predicate)}</span> ${name(rel.target)}</td><td>${esc(rel.conditions || '')}</td></tr>`).join('')).join('')}</tbody></table>`;
  }

  function relateHtml(id) {
    let result = '';
    if (state.to && state.to !== id) {
      const found = paths(id, state.to);
      result = found.length ? found.map((p, i) => `<div class="bok-path"><p class="bok-text-label">${i ? 'Another route' : 'Shortest chain'} · ${p.steps.length} step${p.steps.length > 1 ? 's' : ''}</p><ol>${p.steps.map((s) =>
        `<li>${conceptLink(s.from)} <span class="bok-verb">${esc(verb(s.rel, s.rel.source === s.from ? 'out' : 'in'))}</span> ${conceptLink(s.to)}<p class="bok-cond">${esc(s.rel.conditions || '')}</p></li>`).join('')}</ol></div>`).join('')
        : '<p class="bok-text">No chain of relationships links these concepts yet.</p>';
    }
    return `<section class="bok-relate"><h3 class="bok-section">How does this relate to…</h3>
      <div class="bok-picker"><label class="bok-sr" for="bok-relate">Another concept</label>
        <input id="bok-relate" type="search" placeholder="Another concept, e.g. tail latency" value="${esc(state.to ? concepts.get(state.to)?.label : '')}" autocomplete="off">
        <ul class="bok-results" id="bok-relate-list" role="listbox" hidden></ul></div>${result}</section>`;
  }

  function renderConcept() {
    const id = state.concept;
    const c = concepts.get(id);
    const row = homeOf.get(id);
    const area = row && areaOf.get(row.id);
    if (visited[visited.length - 1] !== id) visited.push(id);
    renderTree(area?.id, row?.id);
    crumbs(area ? [`<a href="#area=${esc(area.id)}">${esc(area.code || '')} ${esc(area.label)}</a>`, `<a href="${subjectHref(row)}">${esc(row.code || '')} ${esc(row.label)}</a>`, `<span aria-current="page">${esc(c.label)}</span>`]
      : [`<span aria-current="page">${esc(c.label)}</span>`]);
    const also = (secondaryOf.get(id) || []).map((r) => subjectLink(r)).join('<br>');
    const passages = (linksOf.get(id) || []).map((l) => `<tr><td>${esc(VOLUME[l.volume] || l.volume)}</td><td><a href="${esc(l.url)}">${esc(l.chapter_title)}</a></td><td>${esc(l.label)}</td></tr>`).join('');
    const chapters = (row?.book_chapters || []).map((ch) => `<tr><td>${esc(VOLUME[ch.volume] || ch.volume)}</td><td><a href="${esc(ch.url)}">${esc(ch.title)}</a></td><td>Teaches ${esc(row.label)}</td></tr>`).join('');
    const links = (outgoing.get(id)?.length || 0) + (incoming.get(id)?.length || 0);
    $('bok-entry-body').innerHTML = `<div class="bok-entry-head">
        <div><h2>${esc(c.label)}</h2>
          ${c.aliases?.length ? `<p class="bok-aliases">Also known as ${c.aliases.map(esc).join(', ')}</p>` : ''}
          <p class="bok-lead">${esc(c.definition)}</p>
          ${c.scope === 'background' ? '<p class="bok-text">A background concept, from an adjacent discipline that ML systems build on.</p>' : ''}</div>
        ${facts([['Identifier', `<code>${esc(id)}</code>`], ['Subject', row ? subjectLink(row) : ''], ['Area', area ? `<a href="#area=${esc(area.id)}">${esc(area.code || '')} ${esc(area.label)}</a>` : ''],
          ['Also part of', also], ['Relationships', String(links)]])}
      </div>
      <figure class="bok-mapbox"><div class="bok-canvas" id="bok-canvas" role="img" aria-label="Concept map of ${esc(c.label)}; the relationships table lists the same links as text"></div>
        <figcaption>What acts on ${esc(c.label)} (left) and what it acts on (right). Select a concept to move to it.</figcaption></figure>
      ${relationshipTable(id)}
      ${passages || chapters ? `<h3 class="bok-section">In the books</h3><table class="bok-table"><thead><tr><th>Volume</th><th>Chapter</th><th>${passages ? 'Passage' : 'Scope'}</th></tr></thead><tbody>${passages || chapters}</tbody></table>` : ''}
      ${relateHtml(id)}
      ${visited.length > 1 ? `<p class="bok-trail"><span>Your path</span>${visited.slice(-8).map((x) => conceptLink(x)).join('<span aria-hidden="true">→</span>')}</p>` : ''}
      <p class="bok-note">Relationships are editorial drafts, each holding under its stated condition. Book links do not assess teaching depth.</p>`;
    wirePicker('bok-relate', 'bok-relate-list', (r) => { location.hash = `${conceptHref(id)}&to=${encodeURIComponent(r.id)}`; }, ['Concept']);
    drawMap(id, $('bok-canvas'));
  }

  function renderConceptLanding() {
    renderTree(null, null);
    crumbs(['<span aria-current="page">Concept map</span>']);
    const central = [...concepts.keys()].filter((id) => concepts.get(id).scope !== 'background')
      .sort((a, b) => (neighbors.get(b)?.length || 0) - (neighbors.get(a)?.length || 0)).slice(0, 15);
    $('bok-entry-body').innerHTML = `<div class="bok-entry-head"><div><h2>Concept map</h2>
      <p class="bok-lead">Every concept has its own page with a map of what acts on it and what it acts on, each link labeled and qualified. Search above, open a concept from any subject, or start with one of the most connected concepts.</p></div></div>
      ${conceptTable(central, 'Most connected concepts')}`;
  }

  // ------------------------------------------------------------------ glossary
  function renderGlossary() {
    const sorted = [...concepts.values()].sort((a, b) => a.label.localeCompare(b.label, undefined, { sensitivity: 'base' }));
    const byLetter = new Map();
    for (const c of sorted) push(byLetter, /[a-z]/i.test(c.label[0]) ? c.label[0].toUpperCase() : '#', c);
    const options = areas.map((a) => `<option value="${esc(a.id)}">${esc(a.code ? `${a.code} ` : '')}${esc(a.label)}</option>`).join('');
    $('bok-glossary').innerHTML = `<div class="bok-entry-head"><div><h2>Glossary</h2><p class="bok-lead">Every concept in the taxonomy with its definition and the subject it belongs to.</p></div></div>
      <div class="bok-gloss-bar">
        <label class="bok-sr" for="bok-gloss-filter">Filter the glossary</label>
        <input id="bok-gloss-filter" type="search" placeholder="Filter ${concepts.size} terms" autocomplete="off">
        <label class="bok-sr" for="bok-gloss-area">Area</label>
        <select id="bok-gloss-area"><option value="">All areas</option>${options}</select>
        <span class="bok-gloss-count" id="bok-gloss-count" aria-live="polite"></span>
      </div>
      <nav class="bok-jump" aria-label="Jump to letter">${[...byLetter.keys()].map((l) => `<a href="#" data-letter="${esc(l)}">${esc(l)}</a>`).join('')}</nav>
      <table class="bok-table bok-gloss"><thead><tr><th>Term</th><th>Definition</th><th>Subject</th></tr></thead>
      ${[...byLetter.entries()].map(([letter, list]) => `<tbody id="letter-${esc(letter)}"><tr class="bok-letter-row"><th colspan="3">${esc(letter)}</th></tr>${list.map((c) => {
        const row = homeOf.get(c.id);
        return `<tr class="bok-term" id="term-${esc(c.id)}" data-area="${esc(areaOf.get(row?.id)?.id || '')}" data-search="${esc([c.label, ...(c.aliases || []), c.definition].join(' ').toLowerCase())}">
          <td>${conceptLink(c.id)}${c.aliases?.length ? `<span class="bok-alias">${c.aliases.map(esc).join(' · ')}</span>` : ''}</td>
          <td>${esc(c.definition)}</td><td>${row ? `<a class="bok-mono" href="${subjectHref(row)}" title="${esc(row.label)}">${esc(row.code || row.label)}</a>` : ''}</td></tr>`;
      }).join('')}</tbody>`).join('')}</table>`;
  }

  function filterGlossary() {
    const q = $('bok-gloss-filter').value.trim().toLowerCase();
    const area = $('bok-gloss-area').value;
    let shown = 0;
    for (const tr of document.querySelectorAll('.bok-term')) {
      const hit = (!q || tr.dataset.search.includes(q)) && (!area || tr.dataset.area === area);
      tr.hidden = !hit;
      shown += hit;
    }
    for (const body of document.querySelectorAll('.bok-gloss tbody')) body.hidden = !body.querySelector('.bok-term:not([hidden])');
    for (const link of document.querySelectorAll('.bok-jump a')) link.classList.toggle('is-empty', !document.getElementById(`letter-${link.dataset.letter}`)?.querySelector('.bok-term:not([hidden])'));
    $('bok-gloss-count').textContent = `${shown} of ${concepts.size} terms`;
  }

  // -------------------------------------------------------------------- search
  function search(query, kinds = ['Concept', 'Subject', 'Area']) {
    const q = query.trim().toLowerCase();
    if (q.length < 2) return [];
    const scored = [];
    const score = (text, weight) => {
      const t = String(text || '').toLowerCase();
      if (!t) return 0;
      if (t === q) return 100 * weight;
      if (t.startsWith(q)) return 60 * weight;
      if (t.split(/[^a-z0-9]+/).some((w) => w.startsWith(q))) return 40 * weight;
      return t.includes(q) ? 20 * weight : 0;
    };
    if (kinds.includes('Concept')) for (const c of concepts.values()) {
      const s = Math.max(score(c.label, 1.2), ...(c.aliases || []).map((a) => score(a, 1.1)), score(c.definition, 0.3));
      if (s) scored.push({ s, kind: 'Concept', id: c.id, label: c.label, where: homeOf.get(c.id)?.code || '' });
    }
    if (kinds.includes('Subject')) for (const r of rows.values()) {
      const s = Math.max(score(r.label, 1), score(r.code, 1.5), score(r.definition, 0.25));
      if (s) scored.push({ s, kind: 'Subject', id: r.id, label: r.label, where: r.code || '' });
    }
    if (kinds.includes('Area')) for (const a of areas) {
      const s = Math.max(score(a.label, 1.3), score(a.code, 1.5));
      if (s) scored.push({ s, kind: 'Area', id: a.id, label: a.label, where: a.code || '' });
    }
    return scored.sort((a, b) => b.s - a.s || a.label.localeCompare(b.label)).slice(0, 10);
  }

  function wirePicker(inputId, listId, onPick, kinds) {
    const input = $(inputId);
    const list = $(listId);
    let found = [];
    let active = 0;
    const show = () => {
      list.innerHTML = found.map((r, i) => `<li role="option" aria-selected="${i === active}" data-i="${i}"><span class="bok-res-kind">${r.kind}</span><span class="bok-res-label">${esc(r.label)}</span><span class="bok-res-where">${esc(r.where)}</span></li>`).join('')
        || (input.value.trim().length >= 2 ? '<li class="bok-noresult">No matches</li>' : '');
      list.hidden = !list.innerHTML;
      input.setAttribute('aria-expanded', String(!list.hidden));
    };
    input.addEventListener('input', () => { found = search(input.value, kinds); active = 0; show(); });
    input.addEventListener('keydown', (e) => {
      if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') && found.length) {
        e.preventDefault();
        active = (active + (e.key === 'ArrowDown' ? 1 : found.length - 1)) % found.length;
        show();
      } else if (e.key === 'Enter' && found[active]) {
        e.preventDefault();
        list.hidden = true;
        input.blur();
        onPick(found[active]);
      } else if (e.key === 'Escape') {
        input.value = '';
        found = [];
        show();
      }
    });
    input.addEventListener('blur', () => setTimeout(() => { list.hidden = true; }, 150));
    list.addEventListener('mousedown', (e) => {
      const item = e.target.closest('[data-i]');
      if (!item) return;
      e.preventDefault();
      list.hidden = true;
      input.blur();
      onPick(found[Number(item.dataset.i)]);
    });
  }

  function choose(r) {
    if (r.kind === 'Concept') location.hash = conceptHref(r.id);
    else if (r.kind === 'Subject') location.hash = subjectHref(rows.get(r.id));
    else location.hash = `#area=${encodeURIComponent(r.id)}`;
  }

  // ---------------------------------------------------------------- navigation
  function restore() {
    const p = new URLSearchParams(location.hash.slice(1));
    const view = p.get('view');
    // Retired identities redirect to their replacement, so old links keep working.
    const replaced = (data.retired || []).find((r) => r.id === p.get('concept'))?.replaced_by;
    const concept = concepts.has(p.get('concept')) ? p.get('concept') : replaced || null;
    const subject = [p.get('subject'), p.get('row')].find((id) => rows.has(id)) || null;
    const area = areas.some((a) => a.id === p.get('area')) ? p.get('area') : null;
    Object.assign(state, { area, subject, concept, to: concepts.has(p.get('to')) ? p.get('to') : null, term: concepts.has(p.get('term')) ? p.get('term') : null });
    if (view === 'glossary' || view === 'index') state.view = 'glossary';
    else if (concept) state.view = 'concept';
    else if (view === 'concept' || view === 'graph') state.view = 'landing';
    else if (subject) state.view = 'subject';
    else if (area) state.view = 'area';
    else state.view = 'home';
    render();
  }

  function render() {
    const v = state.view;
    const entry = ['area', 'subject', 'concept', 'landing'].includes(v);
    $('bok-home').hidden = v !== 'home';
    $('bok-entry').hidden = !entry;
    $('bok-glossary').hidden = v !== 'glossary';
    for (const [tab, views] of [['bok-tab-home', ['home', 'area', 'subject']], ['bok-tab-concept', ['concept', 'landing']], ['bok-tab-glossary', ['glossary']]]) {
      if (views.includes(v)) $(tab).setAttribute('aria-current', 'page');
      else $(tab).removeAttribute('aria-current');
    }
    if (v === 'area') renderArea();
    else if (v === 'subject') renderSubject();
    else if (v === 'concept') renderConcept();
    else if (v === 'landing') renderConceptLanding();
    else if (v === 'glossary') filterGlossary();
    const target = v === 'glossary' && state.term ? document.getElementById(`term-${state.term}`) : null;
    if (target) {
      target.scrollIntoView({ block: 'center', behavior: 'smooth' });
      target.classList.add('is-focus');
      setTimeout(() => target.classList.remove('is-focus'), 1600);
    } else {
      const top = document.querySelector('.bok-toolbar');
      if (top && top.getBoundingClientRect().top < 0) top.scrollIntoView({ block: 'start' });
    }
  }

  function printOutline() {
    document.body.classList.add('bok-printing');
    window.print();
    document.body.classList.remove('bok-printing');
  }

  function bind() {
    window.addEventListener('hashchange', restore);
    $('bok-tab-home').addEventListener('click', (e) => { e.preventDefault(); history.pushState(null, '', location.pathname + location.search); restore(); });
    $('bok-tab-concept').addEventListener('click', (e) => {
      if (!visited.length) return;
      e.preventDefault();
      location.hash = conceptHref(visited[visited.length - 1]);
    });
    $('bok-lenses').addEventListener('click', (e) => {
      const button = e.target.closest('button[data-lens]');
      if (!button) return;
      state.lens = state.lens === button.dataset.lens ? null : button.dataset.lens;
      renderMatrix();
    });
    $('bok-show-concepts').addEventListener('click', () => { state.showConcepts = !state.showConcepts; renderMatrix(); });
    $('bok-gloss-filter').addEventListener('input', filterGlossary);
    $('bok-gloss-area').addEventListener('input', filterGlossary);
    $('bok-print').addEventListener('click', (e) => { e.preventDefault(); printOutline(); });
    document.addEventListener('click', (e) => {
      const letter = e.target.closest('a[data-letter]');
      if (!letter) return;
      e.preventDefault();
      document.getElementById(`letter-${letter.dataset.letter}`)?.scrollIntoView({ block: 'start', behavior: 'smooth' });
    });
    wirePicker('bok-search', 'bok-results', choose);
    document.addEventListener('keydown', (e) => {
      if (e.key === '/' && !/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName)) {
        e.preventDefault();
        $('bok-search').focus();
        $('bok-search').select();
      }
    });
    let width = window.innerWidth;
    window.addEventListener('resize', () => {
      if (state.view !== 'concept' || window.innerWidth === width) return;
      width = window.innerWidth;
      drawMap(state.concept, $('bok-canvas'));
    });
    let dark = document.body.classList.contains('quarto-dark');
    new MutationObserver(() => {
      if (dark === document.body.classList.contains('quarto-dark')) return;
      dark = !dark;
      if (!cy) return;
      readPalette();
      cy.style(mapStyle());
    }).observe(document.body, { attributes: true, attributeFilter: ['class'] });
  }

  load();
})();
