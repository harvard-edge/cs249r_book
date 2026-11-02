// ML Systems Taxonomy, presented the way established taxonomies are (MITRE
// ATT&CK/ATLAS, SFIA, ESCO): a concept-map landing view, the whole taxonomy as
// one matrix (#view=matrix), and an entry page per area, subject, and concept
// with a sidebar tree, a facts box, and tables. Every view has a stable hash:
//   # / #view=map             concept map     #area=<id>          area page
//   #subject=<id>             subject page    #concept=<id>[&to=] concept page
//   #view=matrix              matrix          #overlay=role:<id>|path:<id>|vol:<id>
//   #view=glossary[&term=<id>]                #bok-how-to-use / #bok-release
//   #view=guides[&role=|&symptom=|&path=|&section=<id>]  roles, symptoms, learning paths
// Older hashes (#row=, #view=concept, #view=graph…, #view=index) still resolve.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const push = (map, key, value) => (map.get(key) || map.set(key, []).get(key)).push(value);
  const cap = (text) => text.charAt(0).toUpperCase() + text.slice(1);
  const VOLUME = { vol1: 'Vol I', vol2: 'Vol II', vol3: 'Vol III', vol4: 'Vol IV' };
  const BAND_ORDER = ['lifecycle', 'stack', 'qualities', 'foundations', 'system-types'];
  // Short pill labels for context facets from data.framework.dimensions.contexts.
  const LENS_SHORT_LABELS = {
    generative: 'Generative',
    agentic: 'Agentic',
    physical: 'Physical',
    edge: 'Edge',
    distributed: 'Distributed',
    recommendation: 'Recommendation',
    multimodal: 'Multimodal',
    'cloud-datacenter': 'Cloud / DC',
    'on-device': 'On-device',
    'safety-critical': 'Safety-critical',
    regulated: 'Regulated',
    'multi-tenant': 'Multi-tenant',
    scientific: 'Scientific',
  };
  const MAP_LIMIT = 6;

  let data, groups, areas, rows, concepts, homeOf, areaOf, outgoing, incoming, secondaryOf, linksOf, gapsOf, relations, neighbors, lenses;
  const state = {
    view: 'landing', area: null, subject: null, concept: null, to: null, term: null, guide: null,
    lens: null, overlay: null, pathStep: 0, showConcepts: false,
    couplingMode: 'wheel', couplingFamily: 'all', couplingSel: null,
    activeRole: null, activeAreaRole: null, activeSymptom: null, activeDiagNode: null,
    vizFull: null,
  };
  const ICON_EXPAND = 'M2 6V2h4M10 2h4v4M14 10v4h-4M6 14H2v-4';
  const ICON_EXIT = 'M6 2v4H2M14 6h-4V2M10 14v-4h4M2 10h4v4';
  const expandBtn = (figId) => `<button type="button" class="bok-viz-expand" data-viz-expand="${esc(figId)}" aria-pressed="false">
    <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><path d="${ICON_EXPAND}" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
    <span class="bok-viz-expand-label">Expand to full screen</span></button>`;
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
    renderOverlays();
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
    const rawContexts = fw.dimensions?.contexts || [];
    const primaryOrder = Object.keys(LENS_SHORT_LABELS);
    lenses = rawContexts
      .filter((c) => used.has(c.id))
      .sort((a, b) => {
        const ia = primaryOrder.indexOf(a.id);
        const ib = primaryOrder.indexOf(b.id);
        return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
      })
      .map((c) => [c.id, LENS_SHORT_LABELS[c.id] || c.label.replace(/\s+(?:systems|services|settings)$/i, '')]);
  }

  const groupOf = (area) => groups.find((g) => g.id === area.group);
  const conceptCount = (area) => area.rows.reduce((n, r) => n + r.concepts.length, 0);
  const code = (value) => (value ? `<span class="bok-code">${esc(value)}</span>` : '');
  const subjectHref = (row) => `#subject=${encodeURIComponent(row.id)}`;
  const conceptHref = (id) => `#concept=${encodeURIComponent(id)}`;
  const conceptLink = (id) => `<a href="${conceptHref(id)}"${concepts.get(id)?.scope === 'background' ? ' class="is-background"' : ''}>${esc(concepts.get(id)?.label || id)}</a>`;
  const subjectLink = (row) => `<a href="${subjectHref(row)}">${code(row.code)}${esc(row.label)}</a>`;
  const verb = (rel, direction = 'out') => {
    const r = relations.get(rel.relation);
    return r ? (direction === 'out' ? r.forward : r.inverse) : rel.predicate;
  };
  const crossStepRelationships = (stepIndex) => {
    const matches = [];
    for (const rel of data.relationships || []) {
      const u = homeOf.get(rel.source);
      const v = homeOf.get(rel.target);
      if (!u || !v || u.id === v.id) continue;
      if (!stepIndex.has(u.id) || !stepIndex.has(v.id)) continue;
      matches.push({ rel, iu: stepIndex.get(u.id), iv: stepIndex.get(v.id) });
    }
    return matches;
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
    $('bok-method').textContent = `Version ${fw.version}. Validated against taxonomy/schema/mlsys-taxonomy.linkml.yml (LinkML) and generated from the single canonical source taxonomy/ml-systems-taxonomy.yml.`;
  }

  // -------------------------------------------------------------------- matrix
  // One band per group; each area is a column and each subject a cell.
  function renderLenses() {
    $('bok-lenses').innerHTML = '<span class="bok-filter-label">Highlight</span>' + lenses.map(([id, label]) =>
      `<button type="button" data-lens="${esc(id)}" aria-pressed="false">${esc(label)}</button>`).join('');
  }

  function renderOverlays() {
    const el = $('bok-overlays');
    if (!el) return;
    const g = data.framework.guides || { roles: [], learning_paths: [] };
    const roleOpts = (g.roles || []).map((r) => `<option value="role:${esc(r.id)}">Role: ${esc(r.label)}</option>`).join('');
    const pathOpts = (g.learning_paths || []).map((p) => `<option value="path:${esc(p.id)}">Path: ${esc(p.label)}</option>`).join('');
    const volOpts = Object.entries(VOLUME).map(([k, v]) => `<option value="vol:${esc(k)}">Book: ${esc(v)}</option>`).join('');
    el.innerHTML = `<span class="bok-filter-label">Overlay</span><select id="bok-overlay-select" aria-label="Annotate matrix with a role, learning path, or volume">
      <option value="">None</option>
      ${roleOpts ? `<optgroup label="By role (● Core · ○ Working)">${roleOpts}</optgroup>` : ''}
      ${pathOpts ? `<optgroup label="Learning paths (Steps 1..N)">${pathOpts}</optgroup>` : ''}
      <optgroup label="MLSysBook volume">${volOpts}</optgroup>
    </select>`;
  }

  function overlayCellState(row) {
    if (!state.overlay) return { hit: true, active: false, badge: '' };
    const g = data.framework.guides || { roles: [], learning_paths: [] };
    const [kind, id] = state.overlay.split(':');
    if (kind === 'role') {
      const role = (g.roles || []).find((r) => r.id === id);
      if (!role) return { hit: true, active: false, badge: '' };
      if ((role.core || []).includes(row.id)) {
        return { hit: true, active: false, badge: '<span class="bok-cell-mark-core" title="Core subject for this role" aria-label="Core subject"></span>' };
      }
      if ((role.working || []).includes(row.id)) {
        return { hit: true, active: false, badge: '<span class="bok-cell-mark-working" title="Working knowledge for this role" aria-label="Working subject"></span>' };
      }
      return { hit: false, active: false, badge: '' };
    }
    if (kind === 'path') {
      const lp = (g.learning_paths || []).find((p) => p.id === id);
      if (!lp) return { hit: true, active: false, badge: '' };
      const idx = (lp.steps || []).findIndex((s) => s.subject === row.id);
      if (idx === -1) return { hit: false, active: false, badge: '' };
      const isActive = idx === state.pathStep;
      return {
        hit: true,
        active: isActive,
        badge: `<span class="bok-cell-step" title="Step ${idx + 1} of ${lp.steps.length}">${idx + 1}</span>`,
      };
    }
    if (kind === 'vol') {
      const hasVol = (row.book_volumes || []).includes(id);
      const shortVol = id.replace('vol', 'V');
      return {
        hit: hasVol,
        active: false,
        badge: hasVol ? `<span class="bok-cell-vol">${esc(shortVol)}</span>` : '',
      };
    }
    return { hit: true, active: false, badge: '' };
  }

  function renderOverlayBanner() {
    const banner = $('bok-overlay-banner');
    if (!banner) return;
    const sel = $('bok-overlay-select');
    if (sel) {
      sel.value = state.overlay || '';
      sel.classList.toggle('is-active', Boolean(state.overlay));
    }
    if (!state.overlay) {
      banner.hidden = true;
      banner.innerHTML = '';
      return;
    }
    const g = data.framework.guides || { roles: [], learning_paths: [] };
    const [kind, id] = state.overlay.split(':');
    if (kind === 'role') {
      const role = (g.roles || []).find((r) => r.id === id);
      if (!role) { banner.hidden = true; return; }
      banner.hidden = false;
      banner.innerHTML = `<div class="bok-overlay-head">
        <div><span class="bok-overlay-title">${esc(role.label)}</span><span class="bok-overlay-desc">${esc(role.summary || '')}</span></div>
        <div class="bok-overlay-meta">
          <span><span class="bok-legend-dot"></span>Core (${(role.core || []).length})</span>
          <span><span class="bok-legend-ring"></span>Working (${(role.working || []).length})</span>
          <a href="#view=guides&amp;role=${esc(role.id)}">Open in Guides &rarr;</a>
          <button type="button" class="bok-overlay-btn" data-clear-overlay="1">Clear</button>
        </div>
      </div>`;
      return;
    }
    if (kind === 'path') {
      const lp = (g.learning_paths || []).find((p) => p.id === id);
      if (!lp || !lp.steps?.length) { banner.hidden = true; return; }
      const stepIdx = Math.max(0, Math.min(lp.steps.length - 1, state.pathStep || 0));
      state.pathStep = stepIdx;
      const cur = lp.steps[stepIdx];
      const curRow = rows.get(cur.subject);
      const strip = lp.steps.map((st, i) => {
        const r = rows.get(st.subject);
        return `<button type="button" role="tab" aria-selected="${String(i === stepIdx)}" class="bok-step-node${i === stepIdx ? ' is-active' : ''}" data-path-step="${i}" title="Step ${i + 1}: ${esc(r?.label || st.subject)}">
          <span class="bok-step-circle">${i + 1}</span>
          <span class="bok-step-code">${esc(r?.code || st.subject)}</span>
        </button>`;
      }).join('');
      banner.hidden = false;
      banner.innerHTML = `<div class="bok-overlay-head">
        <div><span class="bok-overlay-title">Learning path: ${esc(lp.label)}</span><span class="bok-overlay-desc">${esc(lp.audience || '')}</span></div>
        <div class="bok-overlay-meta">
          <a href="#view=guides&amp;path=${esc(lp.id)}">Open in Guides &rarr;</a>
          <button type="button" class="bok-overlay-btn" data-clear-overlay="1">Clear</button>
        </div>
      </div>
      <div class="bok-step-strip" role="tablist" aria-label="Steps in ${esc(lp.label)}">${strip}</div>
      <div class="bok-step-detail">
        <div><strong>Step ${stepIdx + 1} of ${lp.steps.length}:</strong> ${curRow ? subjectLink(curRow) : esc(cur.subject)} — <span class="bok-muted">${esc(cur.why || '')}</span></div>
        <div class="bok-step-detail-nav">
          <button type="button" class="bok-overlay-btn" data-step-delta="-1"${stepIdx === 0 ? ' disabled' : ''}>&larr; Prev</button>
          <button type="button" class="bok-overlay-btn" data-step-delta="1"${stepIdx === lp.steps.length - 1 ? ' disabled' : ''}>Next &rarr;</button>
        </div>
      </div>`;
      return;
    }
    if (kind === 'vol') {
      const label = VOLUME[id] || id;
      let count = 0;
      for (const r of rows.values()) if ((r.book_volumes || []).includes(id)) count++;
      banner.hidden = false;
      banner.innerHTML = `<div class="bok-overlay-head">
        <div><span class="bok-overlay-title">MLSysBook ${esc(label)}</span><span class="bok-overlay-desc">${count} of ${rows.size} subjects have linked chapters in ${esc(label)}.</span></div>
        <div class="bok-overlay-meta"><button type="button" class="bok-overlay-btn" data-clear-overlay="1">Clear</button></div>
      </div>`;
    }
  }

  function renderMatrix() {
    renderOverlayBanner();
    const bands = bandOrder().map((id) => {
      const group = groups.find((g) => g.id === id);
      const members = areas.filter((a) => a.group === id);
      const columns = members.map((area) => `<div class="bok-col">
          <a class="bok-col-head" href="#area=${esc(area.id)}">${code(area.code)}<span>${esc(area.display_label || area.label)}</span></a>
          <p class="bok-col-count">${area.rows.length} subjects</p>
          <ul>${area.rows.map((row) => {
            const lensHit = !state.lens || (row.contexts || []).includes(state.lens);
            const ov = overlayCellState(row);
            const classes = ['bok-cell', !lensHit ? 'is-dim' : '', state.overlay && !ov.hit ? 'is-overlay-dim' : '', ov.active ? 'is-overlay-active' : ''].filter(Boolean).join(' ');
            const list = state.showConcepts ? `<span class="bok-cell-concepts">${row.concepts.map((cid) => esc(concepts.get(cid)?.label)).join(' · ')}</span>` : '';
            return `<li><a class="${classes}" href="${subjectHref(row)}" title="${esc(row.definition)}"><span class="bok-cell-head">${code(row.code)}${ov.badge}</span><span class="bok-cell-label">${esc(row.display_label || row.label)}</span>${list}</a></li>`;
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
    const citation = `Vijay Janapa Reddi. ML Systems Taxonomy, version ${fw.version}. Machine Learning Systems, 2026. ${data.url}`;
    $('bok-cite-text').textContent = citation;
    $('bok-cite-meta').innerHTML = `Version ${esc(fw.version)} · Taxonomy data licensed <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Editorial draft for expert review · <a href="https://github.com/harvard-edge/cs249r_book/issues/new?template=taxonomy_feedback.yml">Suggest a change on GitHub</a>`;
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
    $('bok-crumbs').innerHTML = [`<a href="#view=matrix">Matrix</a>`, ...parts].join('<span aria-hidden="true">›</span>');
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
    const isActionableKind = (relId) => {
      const k = relations.get(relId)?.kind;
      return k === 'constraint' || k === 'failure' || k === 'mechanism';
    };
    const blocksFor = (list, direction) => {
      const byRelation = new Map();
      for (const rel of list) push(byRelation, rel.relation || rel.predicate, rel);
      return [...byRelation.entries()].sort((a, b) => (isActionableKind(b[0]) - isActionableKind(a[0])) || b[1].length - a[1].length)
        .map(([relation, rels]) => ({ title: relation, rels, direction }));
    };
    const els = [];
    const seen = new Set([id]);
    // Leave at least 24px between the chip and the spine: Cytoscape ignores a
    // taxi turn closer than its 10px minimum and reroutes the edges through
    // the neighbors, stacking their verb labels. Long labels wrap instead.
    const centerWidth = Math.min(2 * SPINE - 48, Math.round(concepts.get(id).label.length * 8.8 + 40));
    const centerHalf = centerWidth / 2;
    const ys = [];
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
          ys.push(y);
          const into = rel.target === id;
          // The outer segment runs from the concept to the spine; the verb sits on it.
          const segment = COLUMN - w / 2 - SPINE;
          els.push({ group: 'edges', classes: into ? 'into' : 'from', data: { id: rel.id, source: rel.source, target: rel.target,
            label: verb(rel), offset: segment / 2, turn: into ? segment : SPINE - centerHalf } });
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
    // Center on the span of the placed concepts, not on the column height: an
    // off-center chip runs its edges through a neighbor's row, so they enter it
    // vertically and their verb labels stack (seen with two-link concepts).
    let centerY = ys.length ? (Math.min(...ys) + Math.max(...ys)) / 2 : 30 + tall / 2 - 20;
    // A neighbor a few pixels off the center line would also be entered
    // vertically; put the center on its row so that edge runs straight.
    const near = ys.find((y) => y !== centerY && Math.abs(y - centerY) < 20);
    if (near !== undefined) centerY = near;
    els.unshift({ group: 'nodes', classes: 'center', position: { x: 0, y: centerY }, data: { id, label: center.label, w: centerWidth, tw: centerWidth - 20 } });
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
      for (const [relation, rels] of byRelation) bucket.push({ relation, rels, direction, lever: direction === 'in' && ['mechanism', 'failure'].includes(relations.get(relation)?.kind) });
    }
    const ordered = [...groupsIn.filter((g) => g.lever), ...groupsIn.filter((g) => !g.lever), ...groupsOut];
    if (!ordered.length) return '';
    // Each row is a complete proposition ("A increases B"), so it reads
    // correctly on its own, on any screen, without headers or arrows.
    const self = `<b>${esc(concepts.get(id).label)}</b>`;
    const name = (other) => (other === id ? self : conceptLink(other));
    return `<h3 class="bok-section">Relationships</h3><table class="bok-table bok-rels"><thead><tr><th>Relationship</th><th>Holds when</th></tr></thead><tbody>${ordered.map((g) =>
      g.rels.map((rel, i) => `<tr${i === 0 ? ' class="bok-group-start"' : ''}><td>${name(rel.source)} <span class="bok-verb">${esc(verb(rel))}</span> ${name(rel.target)}</td><td>${esc(rel.conditions || '')}</td></tr>`).join('')).join('')}</tbody></table>`;
  }

  function renderPathChainSvg(steps) {
    if (!steps || !steps.length) return '';
    const nodes = [steps[0].from, ...steps.map((s) => s.to)];
    const n = nodes.length;
    const boxW = 148;
    const gapW = 104;
    const padX = 16;
    const totalW = padX * 2 + n * boxW + (n - 1) * gapW;
    const h = 64;
    const cy = 34;
    let svg = `<div class="bok-path-chain"><svg width="${totalW}" height="${h}" viewBox="0 0 ${totalW} ${h}" role="group" aria-label="Relationship chain of ${steps.length} steps">`;
    for (let i = 0; i < steps.length; i++) {
      const s = steps[i];
      const x1 = padX + i * (boxW + gapW) + boxW;
      const x2 = x1 + gapW;
      const mx = (x1 + x2) / 2;
      const dirOut = s.rel.source === s.from;
      const vText = verb(s.rel, dirOut ? 'out' : 'in');
      const lineX1 = dirOut ? x1 : x1 + 5;
      const lineX2 = dirOut ? x2 - 5 : x2;
      svg += `<line x1="${lineX1}" y1="${cy}" x2="${lineX2}" y2="${cy}" stroke="var(--bok-ink)" stroke-width="1.25" stroke-opacity="0.55"/>`;
      const arrowX = dirOut ? x2 - 5 : x1 + 5;
      const arrowPath = dirOut ? `M${arrowX - 4},${cy - 3.5} L${arrowX},${cy} L${arrowX - 4},${cy + 3.5}` : `M${arrowX + 4},${cy - 3.5} L${arrowX},${cy} L${arrowX + 4},${cy + 3.5}`;
      svg += `<path d="${arrowPath}" fill="none" stroke="var(--bok-accent)" stroke-width="1.5" stroke-linecap="round"/>`;
      svg += `<text x="${mx}" y="${cy - 7}" text-anchor="middle" font-size="9.5" font-style="italic" fill="var(--bok-muted)">${esc(vText.length > 20 ? vText.slice(0, 19) + '…' : vText)}</text>`;
    }
    for (let i = 0; i < n; i++) {
      const cid = nodes[i];
      const c = concepts.get(cid);
      const r = homeOf.get(cid);
      const x = padX + i * (boxW + gapW);
      const isEndpoint = i === 0 || i === n - 1;
      const label = c?.label || cid;
      const shortLbl = label.length > 19 ? label.slice(0, 18) + '…' : label;
      svg += `<a href="${conceptHref(cid)}"><g class="viz-interactive" transform="translate(${x},${cy - 18})">
        <rect width="${boxW}" height="36" rx="2" fill="var(--bok-surface)" stroke="${isEndpoint ? 'var(--bok-accent)' : 'var(--bok-line-strong)'}" stroke-width="${isEndpoint ? '1.5' : '1'}"/>
        <text x="8" y="13" class="bok-svg-mono" style="font-size:8.5px;fill:var(--bok-faint)">${esc(r?.code || '')}</text>
        <text x="8" y="27" class="bok-svg-label" style="font-size:10.5px;font-weight:${isEndpoint ? '600' : '500'}">${esc(shortLbl)}</text>
      </g></a>`;
    }
    svg += '</svg></div>';
    return svg;
  }

  function relateHtml(id) {
    let result = '';
    if (state.to && state.to !== id) {
      const found = paths(id, state.to);
      result = found.length ? found.map((p, i) => `<div class="bok-path"><p class="bok-text-label">${i ? 'Another route' : 'Shortest chain'} · ${p.steps.length} step${p.steps.length > 1 ? 's' : ''}</p>${i === 0 ? renderPathChainSvg(p.steps) : ''}<ol>${p.steps.map((s) =>
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
    crumbs(area ? [`<a href="#view=concept">Concept map</a>`, `<a href="#area=${esc(area.id)}">${esc(area.code || '')} ${esc(area.label)}</a>`, `<a href="${subjectHref(row)}">${esc(row.code || '')} ${esc(row.label)}</a>`, `<span aria-current="page">${esc(c.label)}</span>`]
      : [`<a href="#view=concept">Concept map</a>`, `<span aria-current="page">${esc(c.label)}</span>`]);
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
        <figcaption>What acts on ${esc(c.label)} (left) and what it acts on (right). Select a concept to move to it, or open the <a href="#view=concept" style="text-decoration:underline">full field coupling map</a>.</figcaption></figure>
      ${relationshipTable(id)}
      ${passages || chapters ? `<h3 class="bok-section">In the books</h3><table class="bok-table"><thead><tr><th>Volume</th><th>Chapter</th><th>${passages ? 'Passage' : 'Scope'}</th></tr></thead><tbody>${passages || chapters}</tbody></table>` : ''}
      ${relateHtml(id)}
      ${visited.length > 1 ? `<p class="bok-trail"><span>Your path</span>${visited.slice(-8).map((x) => conceptLink(x)).join('<span aria-hidden="true">→</span>')}</p>` : ''}
      <p class="bok-note">Relationships are editorial drafts, each holding under its stated condition. Book links do not assess teaching depth.</p>`;
    wirePicker('bok-relate', 'bok-relate-list', (r) => { location.hash = `${conceptHref(id)}&to=${encodeURIComponent(r.id)}`; }, ['Concept']);
    drawMap(id, $('bok-canvas'));
  }

  // --------------------------------------------- field coupling visualizations
  const REL_FAMILIES = [
    ['all', 'All relationships'],
    ['limits', 'Bottlenecks & limits'],
    ['levers', 'Levers & mitigations'],
    ['structure', 'Dependencies & structure'],
    ['eval', 'Verification & trade-offs'],
  ];

  function relInFamily(rel, fam) {
    if (!fam || fam === 'all') return true;
    const r = rel.relation;
    const kind = relations.get(r)?.kind || rel.kind;
    if (fam === 'limits') return kind === 'constraint' || kind === 'failure';
    if (fam === 'levers') return kind === 'mechanism';
    if (fam === 'structure') return kind === 'dependency' || kind === 'composition';
    if (fam === 'eval') return kind === 'evaluation' || kind === 'tradeoff';
    return true;
  }

  function filteredRelationships() {
    return (data.relationships || []).filter((r) => relInFamily(r, state.couplingFamily));
  }

  // Shared classifier so the wheel, DSM, and coupling panel always agree on counts.
  function couplingScope(sel) {
    if (!sel) return null;
    const areaRows = (id) => new Set((areas.find((a) => a.id === id)?.rows || []).map((r) => r.id));
    if (sel.kind === 'subject' && rows.has(sel.id)) return { kind: 'subject', from: new Set([sel.id]) };
    if (sel.kind === 'area') return { kind: 'area', from: areaRows(sel.id) };
    if (sel.kind === 'pair') return { kind: 'pair', from: areaRows(sel.fromArea), to: areaRows(sel.toArea), diagonal: sel.fromArea === sel.toArea };
    return null;
  }

  // Returns 'out' | 'in' | 'between' (between two subjects of the selection) | 'inside' (within one subject) | null
  function classifyRel(rel, scope) {
    if (!scope) return null;
    const u = homeOf.get(rel.source)?.id;
    const v = homeOf.get(rel.target)?.id;
    if (!u || !v) return null;
    const uIn = scope.from.has(u);
    const vIn = scope.from.has(v);
    if (scope.to) {
      if (!uIn || !scope.to.has(v)) return null;
      return scope.diagonal ? (u === v ? 'inside' : 'between') : 'out';
    }
    if (uIn && vIn) return u === v ? 'inside' : 'between';
    return uIn ? 'out' : vIn ? 'in' : null;
  }

  // Uniform cubic B-spline through control points with clamped endpoints.
  function bsplinePath(pts) {
    if (pts.length < 2) return '';
    if (pts.length === 2) return `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}L${pts[1][0].toFixed(1)},${pts[1][1].toFixed(1)}`;
    const p = [pts[0], pts[0], pts[0], ...pts.slice(1, -1), pts[pts.length - 1], pts[pts.length - 1], pts[pts.length - 1]];
    let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
    for (let i = 0; i < p.length - 3; i++) {
      const p0 = p[i], p1 = p[i + 1], p2 = p[i + 2], p3 = p[i + 3];
      const c1x = (2 * p1[0] + p2[0]) / 3;
      const c1y = (2 * p1[1] + p2[1]) / 3;
      const c2x = (p1[0] + 2 * p2[0]) / 3;
      const c2y = (p1[1] + 2 * p2[1]) / 3;
      const ex = (p1[0] + 4 * p2[0] + p3[0]) / 6;
      const ey = (p1[1] + 4 * p2[1] + p3[1]) / 6;
      d += `C${c1x.toFixed(1)},${c1y.toFixed(1)} ${c2x.toFixed(1)},${c2y.toFixed(1)} ${ex.toFixed(1)},${ey.toFixed(1)}`;
    }
    return d;
  }

  function bundleCurve(uPt, uAreaPt, uGrpPt, rootPt, vGrpPt, vAreaPt, vPt, sameArea, sameGroup, beta = 0.85) {
    let raw;
    if (sameArea) raw = [uPt, uAreaPt, vPt];
    else if (sameGroup) raw = [uPt, uAreaPt, uGrpPt, vAreaPt, vPt];
    else raw = [uPt, uAreaPt, uGrpPt, rootPt, vGrpPt, vAreaPt, vPt];
    const k = raw.length - 1;
    const bundled = raw.map((pt, i) => {
      if (i === 0 || i === k) return pt;
      const t = i / k;
      const lx = uPt[0] + t * (vPt[0] - uPt[0]);
      const ly = uPt[1] + t * (vPt[1] - uPt[1]);
      return [beta * pt[0] + (1 - beta) * lx, beta * pt[1] + (1 - beta) * ly];
    });
    return bsplinePath(bundled);
  }

  function polar(cx, cy, r, a) {
    return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  }

  function svgArc(cx, cy, r, a0, a1, ccw = false) {
    const [x0, y0] = polar(cx, cy, r, ccw ? a1 : a0);
    const [x1, y1] = polar(cx, cy, r, ccw ? a0 : a1);
    const large = Math.abs(a1 - a0) > Math.PI ? 1 : 0;
    const sweep = ccw ? 0 : 1;
    return `M${x0.toFixed(1)},${y0.toFixed(1)} A${r},${r} 0 ${large} ${sweep} ${x1.toFixed(1)},${y1.toFixed(1)}`;
  }

  function canonicalOrder() {
    const orderedGroups = bandOrder().map((gid) => groups.find((g) => g.id === gid)).filter(Boolean);
    const orderedAreas = [];
    const orderedSubjects = [];
    for (const g of orderedGroups) {
      for (const a of areas.filter((ar) => ar.group === g.id)) {
        orderedAreas.push(a);
        for (const r of a.rows) orderedSubjects.push(r);
      }
    }
    return { orderedGroups, orderedAreas, orderedSubjects };
  }

  function drawCouplingWheel(stageEl) {
    const { orderedGroups, orderedAreas, orderedSubjects } = canonicalOrder();
    const W = 880, H = 860, cx = 440, cy = 430;
    const R_node = 204;
    const R_area = 216;
    const R_area_lbl = 228;
    const R_group = 242;
    const R_group_lbl = 251;
    const R_subj_lbl = 265;

    // Assign angles around circle with gaps between areas and groups.
    const groupGap = 3.0;
    const areaGap = 0.85;
    let totalUnits = 0;
    let prevG = null, prevA = null;
    for (const s of orderedSubjects) {
      const a = areaOf.get(s.id);
      if (prevG !== null && a.group !== prevG) totalUnits += groupGap;
      else if (prevA !== null && a.id !== prevA) totalUnits += areaGap;
      totalUnits += 1;
      prevG = a.group;
      prevA = a.id;
    }

    const subjAngle = new Map();
    const areaAngles = new Map();
    const groupAngles = new Map();
    let cursor = 0;
    prevG = null;
    prevA = null;
    for (const s of orderedSubjects) {
      const a = areaOf.get(s.id);
      if (prevG !== null && a.group !== prevG) cursor += groupGap;
      else if (prevA !== null && a.id !== prevA) cursor += areaGap;
      const ang = -Math.PI / 2 + ((cursor + 0.5) / totalUnits) * 2 * Math.PI;
      subjAngle.set(s.id, ang);
      push(areaAngles, a.id, ang);
      push(groupAngles, a.group, ang);
      cursor += 1;
      prevG = a.group;
      prevA = a.id;
    }

    const subjPt = new Map();
    const areaPt = new Map();
    const groupPt = new Map();
    const rootPt = [cx, cy];
    for (const s of orderedSubjects) subjPt.set(s.id, polar(cx, cy, R_node, subjAngle.get(s.id)));
    for (const a of orderedAreas) {
      const angs = areaAngles.get(a.id);
      const mid = (angs[0] + angs[angs.length - 1]) / 2;
      areaPt.set(a.id, polar(cx, cy, R_node * 0.62, mid));
    }
    for (const g of orderedGroups) {
      const angs = groupAngles.get(g.id);
      const mid = (angs[0] + angs[angs.length - 1]) / 2;
      groupPt.set(g.id, polar(cx, cy, R_node * 0.28, mid));
    }

    // Aggregate cross-subject edges.
    const rels = filteredRelationships();
    const pairMap = new Map();
    let crossSubjTotal = 0;
    for (const rel of rels) {
      const u = homeOf.get(rel.source);
      const v = homeOf.get(rel.target);
      if (!u || !v || u.id === v.id) continue;
      crossSubjTotal++;
      const key = `${u.id}__${v.id}`;
      if (!pairMap.has(key)) pairMap.set(key, { u, v, count: 0, rels: [] });
      const entry = pairMap.get(key);
      entry.count++;
      entry.rels.push(rel);
    }

    // Determine active selection scope via shared classifier.
    const sel = state.couplingSel;
    const scope = couplingScope(sel);
    const activeSubjects = scope ? scope.from : null;

    const neighborSubjects = new Set();
    const restEdges = [];
    const activeInternalEdges = [];
    const activeInEdges = [];
    const activeOutEdges = [];

    for (const edge of pairMap.values()) {
      const uId = edge.u.id, vId = edge.v.id;
      const uArea = areaOf.get(uId), vArea = areaOf.get(vId);
      const role = scope ? classifyRel(edge.rels[0], scope) : null;
      // Separate bidirectional curves when selected by using beta = 0.78 for incoming vs 0.85 for outgoing.
      const beta = role === 'in' ? 0.78 : 0.85;
      const d = bundleCurve(
        subjPt.get(uId), areaPt.get(uArea.id), groupPt.get(uArea.group), rootPt,
        groupPt.get(vArea.group), areaPt.get(vArea.id), subjPt.get(vId),
        uArea.id === vArea.id, uArea.group === vArea.group, beta
      );
      const w = Math.min(2.6, 0.55 + Math.sqrt(edge.count) * 0.52).toFixed(2);
      let cls = 'viz-bundle-edge';
      let isHigh = false;
      if (scope) {
        if (role === 'out') {
          cls += ' is-out';
          isHigh = true;
          neighborSubjects.add(uId);
          neighborSubjects.add(vId);
        } else if (role === 'in') {
          cls += ' is-in';
          isHigh = true;
          neighborSubjects.add(uId);
          neighborSubjects.add(vId);
        } else if (role === 'between') {
          cls += ' is-internal';
          isHigh = true;
          neighborSubjects.add(uId);
          neighborSubjects.add(vId);
        } else {
          cls += ' is-dim';
        }
      }
      const pathHtml = `<path class="${cls}" d="${d}" stroke-width="${isHigh ? Math.max(1.3, Number(w) * 1.15).toFixed(2) : w}" data-u="${esc(uId)}" data-v="${esc(vId)}"/>`;
      if (role === 'out') activeOutEdges.push(pathHtml);
      else if (role === 'in') activeInEdges.push(pathHtml);
      else if (role === 'between') activeInternalEdges.push(pathHtml);
      else restEdges.push(pathHtml);
    }

    // Group & area arcs.
    let defs = '';
    let arcsHtml = '';
    const halfStep = (Math.PI / totalUnits) * 0.72;
    for (const g of orderedGroups) {
      const angs = groupAngles.get(g.id);
      const a0 = angs[0] - halfStep;
      const a1 = angs[angs.length - 1] + halfStep;
      const mid = (a0 + a1) / 2;
      const bottom = Math.sin(mid) > 0.25;
      const arcId = `bok-grp-arc-${g.id}`;
      defs += `<path id="${arcId}" d="${svgArc(cx, cy, bottom ? R_group_lbl + 5 : R_group_lbl, a0, a1, bottom)}" fill="none"/>`;
      arcsHtml += `<path d="${svgArc(cx, cy, R_group, a0, a1)}" fill="none" stroke="var(--bok-ink)" stroke-width="2"/>`;
      arcsHtml += `<text class="bok-svg-group-label"><textPath href="#${arcId}" startOffset="50%" text-anchor="middle">${esc(g.label)}</textPath></text>`;
    }

    for (const a of orderedAreas) {
      const angs = areaAngles.get(a.id);
      const a0 = angs[0] - halfStep * 0.85;
      const a1 = angs[angs.length - 1] + halfStep * 0.85;
      const mid = (a0 + a1) / 2;
      const [lx, ly] = polar(cx, cy, R_area_lbl, mid);
      const isAreaSel = sel?.kind === 'area' && sel.id === a.id;
      arcsHtml += `<g class="viz-interactive" data-wheel-area="${esc(a.id)}" role="button" tabindex="0" aria-label="Select area ${esc(a.code)} ${esc(a.label)}">
        <path d="${svgArc(cx, cy, R_area, a0, a1)}" fill="none" stroke="${isAreaSel ? 'var(--bok-accent)' : 'var(--bok-line-strong)'}" stroke-width="${isAreaSel ? '3.5' : '2.5'}"/>
        <text x="${lx.toFixed(1)}" y="${(ly + 3).toFixed(1)}" text-anchor="middle" class="bok-svg-mono" style="font-size:9px;font-weight:700;fill:${isAreaSel ? 'var(--bok-accent)' : 'var(--bok-muted)'}">${esc(a.code)}</text>
      </g>`;
    }

    // Subject dots and radial labels.
    let nodesHtml = '';
    for (const s of orderedSubjects) {
      const ang = subjAngle.get(s.id);
      const [nx, ny] = subjPt.get(s.id);
      const [tx0, ty0] = polar(cx, cy, R_node + 3, ang);
      const [tx1, ty1] = polar(cx, cy, R_area - 2, ang);
      const [lx, ly] = polar(cx, cy, R_subj_lbl, ang);
      const deg = (ang * 180) / Math.PI;
      const leftHalf = Math.cos(ang) < 0;
      const rot = leftHalf ? deg + 180 : deg;
      const anchor = leftHalf ? 'end' : 'start';
      const nameStr = s.display_label || s.label;
      const isSelected = activeSubjects?.has(s.id);
      const isNeighbor = neighborSubjects.has(s.id);
      const isDim = activeSubjects && !isSelected && !isNeighbor;
      const cls = ['viz-subj-group', isSelected ? 'is-selected' : '', isNeighbor ? 'is-neighbor' : '', isDim ? 'is-dim' : ''].filter(Boolean).join(' ');
      const labelBody = leftHalf
        ? `<tspan class="viz-subj-name">${esc(nameStr)} </tspan><tspan>${esc(s.code)}</tspan>`
        : `<tspan>${esc(s.code)} </tspan><tspan class="viz-subj-name">${esc(nameStr)}</tspan>`;
      nodesHtml += `<g class="${cls}" data-wheel-subj="${esc(s.id)}" role="button" tabindex="0" aria-label="${esc(s.code)} ${esc(s.label)}">
        <line x1="${tx0.toFixed(1)}" y1="${ty0.toFixed(1)}" x2="${tx1.toFixed(1)}" y2="${ty1.toFixed(1)}" stroke="var(--bok-line)" stroke-width="0.75"/>
        <circle class="viz-subj-node" cx="${nx.toFixed(1)}" cy="${ny.toFixed(1)}" r="${isSelected ? '3.5' : '2.3'}"/>
        <text class="viz-subj-label" transform="translate(${lx.toFixed(1)},${ly.toFixed(1)}) rotate(${rot.toFixed(2)})" dy="0.32em" text-anchor="${anchor}">${labelBody}</text>
      </g>`;
    }

    const showInternalLegend = sel?.kind === 'area' || (sel?.kind === 'pair' && sel.fromArea === sel.toArea);
    const legendHtml = `<g transform="translate(16, 18)" aria-hidden="true">
      <line x1="0" y1="0" x2="18" y2="0" stroke="var(--bok-accent)" stroke-width="2"/>
      <text x="24" y="3" class="bok-svg-label" style="font-size:10px;fill:var(--bok-muted)">Outgoing (acts on)</text>
      <line x1="0" y1="15" x2="18" y2="15" stroke="var(--bok-ink)" stroke-width="1.6" stroke-opacity="0.72"/>
      <text x="24" y="18" class="bok-svg-label" style="font-size:10px;fill:var(--bok-muted)">Incoming (acted on by)</text>
      ${showInternalLegend ? `<line x1="0" y1="30" x2="18" y2="30" stroke="var(--bok-ink)" stroke-width="1.5" stroke-dasharray="4 2.5" stroke-opacity="0.62"/>
      <text x="24" y="33" class="bok-svg-label" style="font-size:10px;fill:var(--bok-muted)">Between subjects in area</text>` : ''}
    </g>`;

    stageEl.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="group" aria-labelledby="wheel-title wheel-desc">
      <title id="wheel-title">Hierarchical edge-bundling map of the ${orderedSubjects.length} ML systems subjects</title>
      <desc id="wheel-desc">Subjects are grouped into ${orderedAreas.length} areas and ${orderedGroups.length} groups around the circle; bundled splines show ${crossSubjTotal} cross-subject propositions across ${pairMap.size} directed subject pairs.</desc>
      <defs>${defs}</defs>
      <g class="viz-layer-edges">${restEdges.join('')}${activeInternalEdges.join('')}${activeInEdges.join('')}${activeOutEdges.join('')}</g>
      <g class="viz-layer-arcs">${arcsHtml}</g>
      <g class="viz-layer-nodes">${nodesHtml}</g>
      ${legendHtml}
    </svg>`;
  }

  function drawCouplingDsm(stageEl) {
    const { orderedGroups, orderedAreas } = canonicalOrder();
    const rels = filteredRelationships();
    const counts = new Map();
    const rowTotals = new Map();
    const colTotals = new Map();
    for (const a of orderedAreas) { rowTotals.set(a.id, 0); colTotals.set(a.id, 0); }

    for (const rel of rels) {
      const u = homeOf.get(rel.source);
      const v = homeOf.get(rel.target);
      if (!u || !v) continue;
      const ua = areaOf.get(u.id), va = areaOf.get(v.id);
      if (!ua || !va) continue;
      const k = `${ua.id}__${va.id}`;
      counts.set(k, (counts.get(k) || 0) + 1);
      if (ua.id !== va.id) {
        rowTotals.set(ua.id, (rowTotals.get(ua.id) || 0) + 1);
        colTotals.set(va.id, (colTotals.get(va.id) || 0) + 1);
      }
    }

    const offDiag = [];
    for (const [k, c] of counts.entries()) {
      const [u, v] = k.split('__');
      if (u !== v && c > 0) offDiag.push(c);
    }
    const maxOff = Math.max(1, ...offDiag);
    const maxMarginal = Math.max(1, ...rowTotals.values(), ...colTotals.values());
    const binOpacities = [0.11, 0.22, 0.36, 0.54, 0.74];
    const binFor = (c) => {
      if (!c) return 0;
      const ratio = Math.log(c + 1) / Math.log(maxOff + 1);
      return Math.min(4, Math.floor(ratio * 4.99));
    };

    const cell = 22;
    const grpGap = 4;
    const x0 = 192, y0 = 160;
    const pos = [];
    let curOffset = 0;
    let prevG = null;
    const groupBounds = [];
    for (let i = 0; i < orderedAreas.length; i++) {
      const a = orderedAreas[i];
      if (prevG !== null && a.group !== prevG) {
        groupBounds.push(curOffset + grpGap / 2);
        curOffset += grpGap;
      }
      pos.push(curOffset);
      curOffset += cell;
      prevG = a.group;
    }
    const gridSpan = curOffset;
    const W = x0 + gridSpan + 92;
    const H = y0 + gridSpan + 84;

    const sel = state.couplingSel;
    let cellsHtml = '';
    for (let r = 0; r < orderedAreas.length; r++) {
      const ar = orderedAreas[r];
      const ry = y0 + pos[r];
      for (let c = 0; c < orderedAreas.length; c++) {
        const ac = orderedAreas[c];
        const cx = x0 + pos[c];
        const cnt = counts.get(`${ar.id}__${ac.id}`) || 0;
        const isDiag = r === c;
        const isSelected = sel?.kind === 'pair' && sel.fromArea === ar.id && sel.toArea === ac.id;
        const bIdx = !isDiag && cnt > 0 ? binFor(cnt) : -1;
        const fill = isDiag ? 'url(#bok-dsm-hatch)' : (cnt > 0 ? 'var(--bok-ink)' : 'var(--bok-band)');
        const fillOp = isDiag ? '1' : (cnt > 0 ? String(binOpacities[bIdx]) : '0.45');
        const showNum = cnt > 0 || isSelected;
        const textFill = !isDiag && bIdx >= 3 ? 'var(--bok-surface)' : 'var(--bok-ink)';
        cellsHtml += `<g class="viz-dsm-cell${isSelected ? ' is-selected' : ''}" data-dsm-from="${esc(ar.id)}" data-dsm-to="${esc(ac.id)}" role="button" tabindex="0" aria-label="${esc(ar.code)} to ${esc(ac.code)}: ${cnt} propositions">
          <title>${esc(ar.code)} (${esc(ar.display_label || ar.label)}) → ${esc(ac.code)} (${esc(ac.display_label || ac.label)}): ${cnt} propositions</title>
          <rect class="viz-dsm-bg" x="${cx}" y="${ry}" width="${cell - 1}" height="${cell - 1}" fill="${fill}" fill-opacity="${fillOp}" stroke="${isSelected ? 'var(--bok-accent)' : 'none'}"/>
          ${showNum ? `<text x="${cx + (cell - 1) / 2}" y="${ry + 14}" text-anchor="middle" class="bok-svg-mono" style="font-size:8.5px;fill:${textFill};font-weight:${isSelected ? '700' : '500'}">${cnt}</text>` : ''}
        </g>`;
      }
    }

    // Group rules.
    let rulesHtml = '';
    for (const b of groupBounds) {
      rulesHtml += `<line x1="${x0 - 178}" y1="${y0 + b - 0.5}" x2="${x0 + gridSpan + 52}" y2="${y0 + b - 0.5}" stroke="var(--bok-line-strong)" stroke-width="1"/>`;
      rulesHtml += `<line x1="${x0 + b - 0.5}" y1="${y0 - 132}" x2="${x0 + b - 0.5}" y2="${y0 + gridSpan + 48}" stroke="var(--bok-line-strong)" stroke-width="1"/>`;
    }

    // Row and column headers + marginal bars.
    let headersHtml = '';
    for (let i = 0; i < orderedAreas.length; i++) {
      const a = orderedAreas[i];
      const p = pos[i];
      const fullLbl = a.display_label || a.label;
      const isRowSel = (sel?.kind === 'pair' && sel.fromArea === a.id) || (sel?.kind === 'area' && sel.id === a.id);
      const isColSel = (sel?.kind === 'pair' && sel.toArea === a.id) || (sel?.kind === 'area' && sel.id === a.id);
      const rTot = rowTotals.get(a.id) || 0;
      const cTot = colTotals.get(a.id) || 0;
      const rBarW = Math.round((rTot / maxMarginal) * 38);
      const cBarH = Math.round((cTot / maxMarginal) * 28);

      headersHtml += `<g class="viz-dsm-label${isRowSel ? ' is-selected' : ''}" data-dsm-area="${esc(a.id)}" role="button" tabindex="0" aria-label="Select area ${esc(a.code)} ${esc(fullLbl)}">
        <text x="${x0 - 8}" y="${y0 + p + 14}" text-anchor="end"><tspan style="font-weight:700">${esc(a.code)}</tspan> <tspan style="font-family:Inter,sans-serif;font-size:9px;fill:var(--bok-muted)">${esc(fullLbl)}</tspan></text>
      </g>`;
      headersHtml += `<g class="viz-dsm-label${isColSel ? ' is-selected' : ''}" data-dsm-area="${esc(a.id)}" role="button" tabindex="0" aria-label="Select area ${esc(a.code)} ${esc(fullLbl)}" transform="translate(${x0 + p + 14},${y0 - 8}) rotate(-90)">
        <text x="0" y="0" text-anchor="start"><tspan style="font-weight:700">${esc(a.code)}</tspan> <tspan style="font-family:Inter,sans-serif;font-size:8.5px;fill:var(--bok-muted)">${esc(fullLbl)}</tspan></text>
      </g>`;
      // Marginal bars: right (outgoing cross-area) and bottom (incoming cross-area)
      headersHtml += `<rect x="${x0 + gridSpan + 8}" y="${y0 + p + 6}" width="${rBarW}" height="8" fill="var(--bok-ink)" fill-opacity="0.32"><title>${esc(a.code)} outgoing to other areas: ${rTot}</title></rect>`;
      headersHtml += `<text x="${x0 + gridSpan + 12 + rBarW}" y="${y0 + p + 13.5}" class="bok-svg-mono" style="font-size:8px;fill:var(--bok-faint)">${rTot}</text>`;
      headersHtml += `<rect x="${x0 + p + 6}" y="${y0 + gridSpan + 8}" width="8" height="${cBarH}" fill="var(--bok-ink)" fill-opacity="0.32"><title>${esc(a.code)} incoming from other areas: ${cTot}</title></rect>`;
      headersHtml += `<text x="${x0 + p + 10}" y="${y0 + gridSpan + 18 + cBarH}" text-anchor="middle" class="bok-svg-mono" style="font-size:7.5px;fill:var(--bok-faint)">${cTot}</text>`;
    }

    stageEl.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="group" aria-labelledby="dsm-title dsm-desc">
      <title id="dsm-title">${orderedAreas.length} by ${orderedAreas.length} Area Design Structure Matrix</title>
      <desc id="dsm-desc">Rows represent source areas and columns represent target areas across ${rels.length} propositions.</desc>
      <defs>
        <pattern id="bok-dsm-hatch" width="5" height="5" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
          <line x1="0" y1="0" x2="0" y2="5" stroke="var(--bok-ink)" stroke-width="1" stroke-opacity="0.22"/>
        </pattern>
      </defs>
      <text x="${x0 - 8}" y="${y0 - 10}" text-anchor="end" class="bok-svg-mono" style="font-size:9px;fill:var(--bok-faint)">row → col</text>
      <text x="${x0 + gridSpan + 8}" y="${y0 - 10}" text-anchor="start" class="bok-svg-mono" style="font-size:8.5px;fill:var(--bok-faint)">Out</text>
      <text x="${x0 - 8}" y="${y0 + gridSpan + 18}" text-anchor="end" class="bok-svg-mono" style="font-size:8.5px;fill:var(--bok-faint)">In</text>
      <g>${rulesHtml}</g>
      <g>${cellsHtml}</g>
      <g>${headersHtml}</g>
    </svg>`;
  }

  function renderCouplingPanel(panelEl) {
    const rels = filteredRelationships();
    const sel = state.couplingSel;
    const famLabel = REL_FAMILIES.find(([k]) => k === state.couplingFamily)?.[1] || 'All relationships';

    const PREVIEW = 25;
    const formatProp = (rel) => {
      const uSubj = homeOf.get(rel.source);
      const vSubj = homeOf.get(rel.target);
      return `<li><span style="white-space:nowrap">${conceptLink(rel.source)} <span class="bok-mono" style="font-size:0.72rem">(${esc(uSubj?.code || '')})</span></span> <span class="bok-verb">${esc(verb(rel))}</span> <span style="white-space:nowrap">${conceptLink(rel.target)} <span class="bok-mono" style="font-size:0.72rem">(${esc(vSubj?.code || '')})</span></span>${rel.conditions ? `<p class="bok-cond">${esc(rel.conditions)}</p>` : ''}</li>`;
    };
    const propList = (list) => !list.length
      ? '<p class="bok-muted" style="margin-top:4px !important">None in current filter.</p>'
      : `<ul class="bok-viz-prop-list">${list.slice(0, PREVIEW).map(formatProp).join('')}</ul>` +
        (list.length > PREVIEW
          ? `<details class="bok-viz-more"><summary>Show the other ${list.length - PREVIEW} of ${list.length} propositions</summary><ul class="bok-viz-prop-list">${list.slice(PREVIEW).map(formatProp).join('')}</ul></details>`
          : '');

    if (!sel) {
      // Compute exact 3-way decomposition and top cross-area couplings for default overview.
      const areaPairs = new Map();
      const subjPairs = new Set();
      let crossAreaTotal = 0, betweenSubjSameArea = 0, sameSubjTotal = 0;
      for (const rel of rels) {
        const u = homeOf.get(rel.source), v = homeOf.get(rel.target);
        if (!u || !v) continue;
        const ua = areaOf.get(u.id), va = areaOf.get(v.id);
        if (!ua || !va) continue;
        if (u.id === v.id) {
          sameSubjTotal++;
        } else {
          subjPairs.add(`${u.id}__${v.id}`);
          if (ua.id === va.id) betweenSubjSameArea++;
          else {
            crossAreaTotal++;
            const k = `${ua.id}__${va.id}`;
            areaPairs.set(k, (areaPairs.get(k) || 0) + 1);
          }
        }
      }
      const crossSubjTotal = crossAreaTotal + betweenSubjSameArea;
      const intraAreaTotal = betweenSubjSameArea + sameSubjTotal;
      const modeNote = state.couplingMode === 'dsm'
        ? `Matrix cells show ${crossAreaTotal} off-diagonal (cross-area) and ${intraAreaTotal} diagonal (within-area) propositions.`
        : `Circular bundle draws the ${crossSubjTotal} cross-subject propositions (${subjPairs.size} directed chords); ${sameSubjTotal} are within a single subject.`;
      const topPairs = [...areaPairs.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10);
      panelEl.innerHTML = `<div class="bok-viz-panel-head">
        <div>
          <div class="bok-viz-panel-title">${esc(famLabel)}</div>
          <p class="bok-viz-panel-sub">${rels.length} propositions = ${crossAreaTotal} cross-area + ${betweenSubjSameArea} between subjects in same area + ${sameSubjTotal} within one subject</p>
          <p class="bok-viz-panel-sub" style="margin-top:2px !important">${esc(modeNote)}</p>
        </div>
      </div>
      <div class="bok-viz-prop-group">
        <p class="bok-viz-prop-label">Strongest cross-area couplings (select to inspect)</p>
        <div>${topPairs.map(([k, cnt]) => {
          const [a1Id, a2Id] = k.split('__');
          const a1 = areas.find((a) => a.id === a1Id), a2 = areas.find((a) => a.id === a2Id);
          return `<button type="button" class="bok-viz-pair-btn" data-select-pair="${esc(k)}">
            <span><span class="bok-code">${esc(a1?.code)} → ${esc(a2?.code)}</span>${esc(a1?.display_label || a1?.label)} → ${esc(a2?.display_label || a2?.label)}</span>
            <span class="bok-mono">${cnt}</span>
          </button>`;
        }).join('')}</div>
      </div>`;
      return;
    }

    const scope = couplingScope(sel);
    if (sel.kind === 'pair') {
      const a1 = areas.find((a) => a.id === sel.fromArea);
      const a2 = areas.find((a) => a.id === sel.toArea);
      if (scope.diagonal) {
        const betweenProps = [];
        const insideProps = [];
        for (const rel of rels) {
          const role = classifyRel(rel, scope);
          if (role === 'between') betweenProps.push(rel);
          else if (role === 'inside') insideProps.push(rel);
        }
        const tot = betweenProps.length + insideProps.length;
        panelEl.innerHTML = `<div class="bok-viz-panel-head">
          <div>
            <div class="bok-viz-panel-title">${esc(a1?.code)} → ${esc(a1?.code)} · Within ${esc(a1?.display_label || a1?.label)}</div>
            <p class="bok-viz-panel-sub">${tot} within-area propositions (${betweenProps.length} between subjects · ${insideProps.length} within a single subject)</p>
          </div>
          <button type="button" class="bok-viz-clear" data-clear-coupling="1">Clear (Esc)</button>
        </div>
        <p style="margin-top:6px !important;font-size:0.76rem"><button type="button" class="bok-viz-clear" data-dsm-area="${esc(a1.id)}">Select ${esc(a1.code)} as full area (outgoing, incoming &amp; within)</button></p>
        <div class="bok-viz-prop-group">
          <p class="bok-viz-prop-label"><span class="bok-viz-swatch-internal"></span>Between subjects in ${esc(a1?.code)} (${betweenProps.length})</p>
          ${propList(betweenProps)}
        </div>
        <div class="bok-viz-prop-group">
          <p class="bok-viz-prop-label">Within a single subject in ${esc(a1?.code)} (${insideProps.length})</p>
          ${propList(insideProps)}
        </div>`;
        return;
      }
      const forward = [];
      const reverse = [];
      for (const rel of rels) {
        const u = homeOf.get(rel.source), v = homeOf.get(rel.target);
        if (!u || !v) continue;
        const ua = areaOf.get(u.id), va = areaOf.get(v.id);
        if (ua?.id === a1?.id && va?.id === a2?.id) forward.push(rel);
        if (ua?.id === a2?.id && va?.id === a1?.id) reverse.push(rel);
      }
      panelEl.innerHTML = `<div class="bok-viz-panel-head">
        <div>
          <div class="bok-viz-panel-title">${esc(a1?.code)} → ${esc(a2?.code)} · ${esc(a1?.display_label || a1?.label)} → ${esc(a2?.display_label || a2?.label)}</div>
          <p class="bok-viz-panel-sub">${esc(a1?.code)} → ${esc(a2?.code)}: ${forward.length} · ${esc(a2?.code)} → ${esc(a1?.code)}: ${reverse.length}</p>
        </div>
        <button type="button" class="bok-viz-clear" data-clear-coupling="1">Clear (Esc)</button>
      </div>
      ${reverse.length ? `<p style="margin-top:6px !important;font-size:0.76rem"><button type="button" class="bok-viz-clear" data-select-pair="${esc(a2.id)}__${esc(a1.id)}">Flip direction to ${esc(a2.code)} → ${esc(a1.code)} (${reverse.length})</button></p>` : ''}
      <div class="bok-viz-prop-group">
        <p class="bok-viz-prop-label"><span class="bok-viz-swatch-out"></span>Propositions (${forward.length})</p>
        ${propList(forward)}
      </div>`;
      return;
    }

    // Subject or Area selection via shared classifier.
    const isSubj = sel.kind === 'subject';
    const targetRow = isSubj ? rows.get(sel.id) : null;
    const targetArea = isSubj ? areaOf.get(sel.id) : areas.find((a) => a.id === sel.id);
    const outProps = [];
    const inProps = [];
    const betweenProps = [];
    const insideProps = [];
    for (const rel of rels) {
      const role = classifyRel(rel, scope);
      if (role === 'out') outProps.push(rel);
      else if (role === 'in') inProps.push(rel);
      else if (role === 'between') betweenProps.push(rel);
      else if (role === 'inside') insideProps.push(rel);
    }

    const title = isSubj ? `${targetRow?.code} ${targetRow?.label}` : `${targetArea?.code} ${targetArea?.label}`;
    const openHref = isSubj ? subjectHref(targetRow) : `#area=${encodeURIComponent(targetArea?.id || '')}`;
    const subSummary = isSubj
      ? `${outProps.length} outgoing · ${inProps.length} incoming · ${insideProps.length} within ${targetRow?.code || 'subject'}`
      : `${outProps.length} outgoing · ${inProps.length} incoming · ${betweenProps.length + insideProps.length} within ${targetArea?.code} (${betweenProps.length} between subjects · ${insideProps.length} inside one subject)`;

    panelEl.innerHTML = `<div class="bok-viz-panel-head">
      <div>
        <div class="bok-viz-panel-title">${esc(title)}</div>
        <p class="bok-viz-panel-sub">${esc(subSummary)} · <a href="${openHref}" style="text-decoration:underline;font-weight:600">Open page &rarr;</a></p>
      </div>
      <button type="button" class="bok-viz-clear" data-clear-coupling="1">Clear (Esc)</button>
    </div>
    <div class="bok-viz-prop-group">
      <p class="bok-viz-prop-label"><span class="bok-viz-swatch-out"></span>Outgoing (${outProps.length}) — acts on other ${isSubj ? 'subjects' : 'areas/subjects'}</p>
      ${propList(outProps)}
    </div>
    <div class="bok-viz-prop-group">
      <p class="bok-viz-prop-label"><span class="bok-viz-swatch-in"></span>Incoming (${inProps.length}) — acted on by other ${isSubj ? 'subjects' : 'areas/subjects'}</p>
      ${propList(inProps)}
    </div>
    ${!isSubj ? `<div class="bok-viz-prop-group">
      <p class="bok-viz-prop-label"><span class="bok-viz-swatch-internal"></span>Between subjects in ${esc(targetArea?.code)} (${betweenProps.length})</p>
      ${propList(betweenProps)}
    </div>` : ''}
    ${insideProps.length ? `<div class="bok-viz-prop-group">
      <p class="bok-viz-prop-label">Within ${isSubj ? esc(targetRow?.code) : 'a single subject'} (${insideProps.length})</p>
      ${propList(insideProps)}
    </div>` : ''}`;
  }

  function refreshCouplingFigure() {
    const stageEl = $('bok-coupling-stage');
    const panelEl = $('bok-coupling-panel');
    const capEl = $('bok-coupling-caption');
    if (!stageEl || !panelEl) return;
    for (const b of document.querySelectorAll('[data-coupling-mode]')) {
      b.setAttribute('aria-pressed', String(b.dataset.couplingMode === state.couplingMode));
    }
    for (const b of document.querySelectorAll('[data-coupling-fam]')) {
      b.setAttribute('aria-pressed', String(b.dataset.couplingFam === state.couplingFamily));
    }
    if (state.couplingMode === 'dsm') {
      drawCouplingDsm(stageEl);
      if (capEl) capEl.innerHTML = '<strong>Figure 1b. Cross-area design structure matrix (row = source area, column = target area).</strong> Off-diagonal cell opacity encodes the number of propositions from concepts in the row area to concepts in the column area across five logarithmic bins; diagonal cells (hatched) count propositions within the same area. Marginal bars show total cross-area outgoing (right) and incoming (bottom) couplings. Select any cell to inspect its propositions and directional asymmetry.';
    } else {
      drawCouplingWheel(stageEl);
      if (capEl) capEl.innerHTML = `<strong>Figure 1a. Hierarchical edge bundling of cross-subject relationships across the ${groups.length} groups, ${areas.length} areas, and ${rows.size} subjects (β = 0.85).</strong> Curve width encodes the square root of the proposition count between two subjects. Select any subject or area arc to isolate its outgoing couplings (crimson) and incoming couplings (ink), and read the exact propositions in the panel.`;
    }
    renderCouplingPanel(panelEl);
  }

  function renderConceptLanding() {
    renderTree(null, null);
    crumbs(['<span aria-current="page">Concept map</span>']);
    const central = [...concepts.keys()].filter((id) => concepts.get(id).scope !== 'background')
      .sort((a, b) => (neighbors.get(b)?.length || 0) - (neighbors.get(a)?.length || 0)).slice(0, 15);
    const lastVisited = visited.length ? visited[visited.length - 1] : null;
    const famBtns = REL_FAMILIES.map(([k, lbl]) =>
      `<button type="button" data-coupling-fam="${esc(k)}" aria-pressed="${String(state.couplingFamily === k)}">${esc(lbl)}</button>`).join('');

    $('bok-entry-body').innerHTML = `<div class="bok-entry-head"><div><h2>Concept map</h2>
      <p class="bok-lead">Every concept is linked to what it depends on, enables, limits, or mitigates. Use the field coupling figure below to see how the ${areas.length} areas and ${rows.size} subjects connect across the taxonomy, or open any concept for its local neighborhood map.</p>
      ${lastVisited && concepts.has(lastVisited) ? `<p class="bok-text" style="margin-top:0.4rem !important">Recently viewed: ${conceptLink(lastVisited)} &rarr;</p>` : ''}
      </div></div>
      <figure class="bok-viz-figure" id="bok-coupling-figure">
        <div class="bok-viz-bar">
          <div class="bok-viz-controls">
            <span class="bok-filter-label">View</span>
            <div class="bok-viz-seg" role="group" aria-label="Coupling diagram type">
              <button type="button" data-coupling-mode="wheel" aria-pressed="${String(state.couplingMode === 'wheel')}">Circular bundle (${rows.size} subjects)</button>
              <button type="button" data-coupling-mode="dsm" aria-pressed="${String(state.couplingMode === 'dsm')}">Area matrix (${areas.length} × ${areas.length} DSM)</button>
            </div>
          </div>
          <div class="bok-viz-controls">
            <span class="bok-filter-label">Relations</span>
            <div class="bok-viz-seg" role="group" aria-label="Filter by relationship family">${famBtns}</div>
          </div>
          ${expandBtn('bok-coupling-figure')}
        </div>
        <div class="bok-viz-split">
          <div class="bok-viz-stage" id="bok-coupling-stage"></div>
          <div class="bok-viz-panel" id="bok-coupling-panel" role="region" aria-label="Selected coupling propositions" aria-live="polite"></div>
        </div>
        <figcaption class="bok-viz-caption" id="bok-coupling-caption"></figcaption>
      </figure>
      ${conceptTable(central, 'Most connected concepts')}`;
    refreshCouplingFigure();
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

  // ------------------------------------------------ guides & flow diagrams
  function sankeyRibbon(x0, y0, x1, y1, h0, h1) {
    const mx = (x0 + x1) / 2;
    return `M${x0.toFixed(1)},${y0.toFixed(1)} C${mx.toFixed(1)},${y0.toFixed(1)} ${mx.toFixed(1)},${y1.toFixed(1)} ${x1.toFixed(1)},${y1.toFixed(1)} L${x1.toFixed(1)},${(y1 + h1).toFixed(1)} C${mx.toFixed(1)},${(y1 + h1).toFixed(1)} ${mx.toFixed(1)},${(y0 + h0).toFixed(1)} ${x0.toFixed(1)},${(y0 + h0).toFixed(1)} Z`;
  }

  function drawRoleSankey() {
    const stage = $('bok-role-sankey-stage');
    const bar = $('bok-role-sankey-status');
    if (!stage) return;
    const g = data.framework.guides || { roles: [] };
    const { orderedGroups, orderedAreas } = canonicalOrder();
    const areaIndex = new Map(orderedAreas.map((a, i) => [a.id, i]));

    // Order roles by weighted barycenter across areas (Core = 2, Working = 1).
    const rolesOrdered = [...(g.roles || [])].map((r) => {
      let wSum = 0, idxSum = 0;
      for (const sid of r.core || []) {
        const a = areaOf.get(sid);
        if (a && areaIndex.has(a.id)) { wSum += 2; idxSum += 2 * areaIndex.get(a.id); }
      }
      for (const sid of r.working || []) {
        const a = areaOf.get(sid);
        if (a && areaIndex.has(a.id)) { wSum += 1; idxSum += 1 * areaIndex.get(a.id); }
      }
      return { role: r, totalW: wSum || 1, bary: wSum ? idxSum / wSum : 0 };
    }).sort((a, b) => a.bary - b.bary);

    // Compute weights for Role -> Group and Group -> Area.
    const rgLinks = [];
    const gaMap = new Map();
    const groupW = new Map(orderedGroups.map((gr) => [gr.id, 0]));
    const areaW = new Map(orderedAreas.map((ar) => [ar.id, 0]));

    for (const { role } of rolesOrdered) {
      const byGrp = new Map();
      const addSubj = (sid, kind, w) => {
        const a = areaOf.get(sid);
        if (!a) return;
        groupW.set(a.group, (groupW.get(a.group) || 0) + w);
        areaW.set(a.id, (areaW.get(a.id) || 0) + w);
        const gk = `${a.group}__${kind}`;
        byGrp.set(gk, (byGrp.get(gk) || 0) + w);
        const gak = `${role.id}__${a.group}__${a.id}__${kind}`;
        gaMap.set(gak, (gaMap.get(gak) || 0) + w);
      };
      for (const sid of role.core || []) addSubj(sid, 'core', 2);
      for (const sid of role.working || []) addSubj(sid, 'working', 1);
      for (const gr of orderedGroups) {
        for (const kind of ['core', 'working']) {
          const w = byGrp.get(`${gr.id}__${kind}`) || 0;
          if (w > 0) rgLinks.push({ roleId: role.id, groupId: gr.id, kind, w });
        }
      }
    }

    const totalWeight = [...groupW.values()].reduce((a, b) => a + b, 0) || 1;
    const W = 920, H = 620;
    const topY = 22, botY = H - 18;
    const availH = botY - topY;
    const xRole = 248, xGrp = 476, xArea = 700;
    const barW = 3;

    const layoutCol = (items, getW, pad, minH = 5) => {
      const n = items.length;
      const totPad = pad * (n - 1);
      const rawHeights = items.map((it) => Math.max(minH, getW(it) * ((availH - totPad) / totalWeight)));
      const rawSum = rawHeights.reduce((a, b) => a + b, 0) || 1;
      const factor = (availH - totPad) / rawSum;
      let y = topY;
      const boxes = new Map();
      items.forEach((it, idx) => {
        const h = Math.max(minH, rawHeights[idx] * factor);
        boxes.set(it.id, { y, h, inY: y, outY: y, item: it });
        y += h + pad;
      });
      return { boxes };
    };

    const roleCol = layoutCol(rolesOrdered.map((x) => ({ id: x.role.id, label: x.role.label, w: x.totalW, role: x.role })), (x) => x.w, 10, 9);
    const grpCol = layoutCol(orderedGroups.map((gr) => ({ id: gr.id, label: gr.label, code: gr.code, w: groupW.get(gr.id) || 1 })), (x) => x.w, 20, 14);
    const areaCol = layoutCol(orderedAreas.map((ar) => ({ id: ar.id, label: ar.display_label || ar.label, code: ar.code, w: areaW.get(ar.id) || 1 })), (x) => x.w, 8, 8);

    const activeRole = state.activeRole;
    const activeArea = state.activeAreaRole;
    const areaShare = new Map();
    if (activeArea) {
      for (const { role } of rolesOrdered) {
        const core = (role.core || []).filter((s) => areaOf.get(s)?.id === activeArea).length;
        const working = (role.working || []).filter((s) => areaOf.get(s)?.id === activeArea).length;
        if (core || working) areaShare.set(role.id, { core, working });
      }
    }

    // Build Role -> Group ribbons.
    let ribbonsHtml = '';
    for (const lnk of rgLinks) {
      const rBox = roleCol.boxes.get(lnk.roleId);
      const gBox = grpCol.boxes.get(lnk.groupId);
      const h0 = (lnk.w / rBox.item.w) * rBox.h;
      const h1 = (lnk.w / gBox.item.w) * gBox.h;
      const y0 = rBox.outY, y1 = gBox.inY;
      const d = sankeyRibbon(xRole + barW, y0, xGrp, y1, h0, h1);
      rBox.outY += h0;
      gBox.inY += h1;
      let cls = `viz-ribbon${lnk.kind === 'working' ? ' is-working' : ''}`;
      if (activeRole) {
        cls += lnk.roleId === activeRole ? ' is-active' : ' is-dim';
        ribbonsHtml += `<path class="${cls}" d="${d}" data-sankey-role="${esc(lnk.roleId)}"/>`;
      } else if (activeArea) {
        ribbonsHtml += `<path class="${cls} is-dim" d="${d}" data-sankey-role="${esc(lnk.roleId)}"/>`;
        const wA = gaMap.get(`${lnk.roleId}__${lnk.groupId}__${activeArea}__${lnk.kind}`) || 0;
        if (wA > 0) {
          const dSub = sankeyRibbon(xRole + barW, y0, xGrp, y1, (h0 * wA) / lnk.w, (h1 * wA) / lnk.w);
          ribbonsHtml += `<path class="viz-ribbon is-active${lnk.kind === 'working' ? ' is-working' : ''}" d="${dSub}" data-sankey-role="${esc(lnk.roleId)}"/>`;
        }
      } else {
        ribbonsHtml += `<path class="${cls}" d="${d}" data-sankey-role="${esc(lnk.roleId)}"/>`;
      }
    }

    // Build Group -> Area ribbons (stacked by canonical area order, then role order).
    for (const gr of orderedGroups) {
      const gBox = grpCol.boxes.get(gr.id);
      for (const ar of orderedAreas.filter((a) => a.group === gr.id)) {
        const aBox = areaCol.boxes.get(ar.id);
        for (const { role } of rolesOrdered) {
          for (const kind of ['core', 'working']) {
            const w = gaMap.get(`${role.id}__${gr.id}__${ar.id}__${kind}`) || 0;
            if (!w) continue;
            const h0 = (w / gBox.item.w) * gBox.h;
            const h1 = (w / aBox.item.w) * aBox.h;
            const d = sankeyRibbon(xGrp + barW, gBox.outY, xArea, aBox.inY, h0, h1);
            gBox.outY += h0;
            aBox.inY += h1;
            let cls = `viz-ribbon${kind === 'working' ? ' is-working' : ''}`;
            if (activeRole) cls += role.id === activeRole ? ' is-active' : ' is-dim';
            else if (activeArea) cls += ar.id === activeArea ? ' is-active' : ' is-dim';
            ribbonsHtml += `<path class="${cls}" d="${d}" data-sankey-role="${esc(role.id)}" data-sankey-area="${esc(ar.id)}"/>`;
          }
        }
      }
    }

    let nodesHtml = `<text x="${xRole}" y="12" text-anchor="end" class="bok-svg-group-label">Roles (${rolesOrdered.length})</text>
      <text x="${xGrp + 1.5}" y="12" text-anchor="middle" class="bok-svg-group-label">Groups (${orderedGroups.length})</text>
      <text x="${xArea + barW}" y="12" text-anchor="start" class="bok-svg-group-label">Taxonomy Areas (${orderedAreas.length})</text>`;

    for (const [id, b] of roleCol.boxes) {
      const isSel = activeRole === id || (activeArea && areaShare.has(id));
      const isDim = (activeRole && activeRole !== id) || (activeArea && !areaShare.has(id));
      const cy = b.y + b.h / 2;
      nodesHtml += `<g class="viz-interactive" data-sankey-role="${esc(id)}" role="button" tabindex="0" aria-label="Role ${esc(b.item.label)}" style="${isDim ? 'opacity:0.36' : ''}">
        <rect x="${xRole}" y="${b.y.toFixed(1)}" width="${barW}" height="${b.h.toFixed(1)}" fill="${isSel ? 'var(--bok-accent)' : 'var(--bok-ink)'}"/>
        <text x="${xRole - 8}" y="${(cy + 3.5).toFixed(1)}" text-anchor="end" class="bok-svg-label" style="font-weight:${isSel ? '700' : '500'};fill:${isSel ? 'var(--bok-accent)' : 'var(--bok-ink)'}">${esc(b.item.label)}</text>
      </g>`;
    }
    for (const [, b] of grpCol.boxes) {
      const cy = b.y + b.h / 2;
      const pillW = b.item.label.length * 7.2 + 14;
      nodesHtml += `<g>
        <rect x="${xGrp}" y="${b.y.toFixed(1)}" width="${barW}" height="${b.h.toFixed(1)}" fill="var(--bok-ink)"/>
        <rect x="${(xGrp - pillW - 5).toFixed(1)}" y="${(cy - 6.5).toFixed(1)}" width="${pillW.toFixed(1)}" height="13" rx="2" fill="var(--bok-surface)" fill-opacity="0.92"/>
        <text x="${xGrp - 9}" y="${(cy + 3.2).toFixed(1)}" text-anchor="end" class="bok-svg-group-label" style="fill:var(--bok-ink)">${esc(b.item.label)}</text>
      </g>`;
    }
    for (const [id, b] of areaCol.boxes) {
      const isSel = activeArea === id;
      const cy = b.y + b.h / 2;
      nodesHtml += `<g class="viz-interactive" data-sankey-area="${esc(id)}" role="button" tabindex="0" aria-label="Area ${esc(b.item.code)} ${esc(b.item.label)}">
        <rect x="${xArea}" y="${b.y.toFixed(1)}" width="${barW}" height="${b.h.toFixed(1)}" fill="${isSel ? 'var(--bok-accent)' : 'var(--bok-ink)'}"/>
        <text x="${xArea + barW + 7}" y="${(cy + 3.2).toFixed(1)}" text-anchor="start" class="bok-svg-label" style="font-size:10px;font-weight:${isSel ? '700' : '400'};fill:${isSel ? 'var(--bok-accent)' : 'var(--bok-ink)'}">
          <tspan class="bok-svg-mono" style="font-weight:700;fill:${isSel ? 'var(--bok-accent)' : 'var(--bok-ink)'}">${esc(b.item.code)}</tspan> ${esc(b.item.label)}
        </text>
      </g>`;
    }

    stage.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="group" aria-labelledby="sankey-title sankey-desc">
      <title id="sankey-title">Alluvial diagram mapping ${rolesOrdered.length} ML systems roles across ${orderedGroups.length} groups and ${orderedAreas.length} areas</title>
      <desc id="sankey-desc">Shows how core (weight 2) and working (weight 1) subject requirements distribute across the taxonomy for each role.</desc>
      <g>${ribbonsHtml}</g>
      <g>${nodesHtml}</g>
    </svg>`;

    if (bar) {
      if (activeRole) {
        const r = (g.roles || []).find((x) => x.id === activeRole);
        bar.innerHTML = `<span>Selected role: <strong>${esc(r?.label || activeRole)}</strong> (${(r?.core || []).length} core, ${(r?.working || []).length} working subjects)</span>
          <span><a href="#overlay=role:${esc(activeRole)}" class="bok-viz-clear" style="margin-right:12px">Highlight on Matrix &nearr;</a><button type="button" class="bok-viz-clear" data-clear-sankey="1">Clear selection</button></span>`;
      } else if (activeArea) {
        const a = areas.find((x) => x.id === activeArea);
        const coreRoles = [...areaShare.values()].filter((v) => v.core > 0).length;
        const workRoles = [...areaShare.values()].filter((v) => v.working > 0).length;
        const bothRoles = [...areaShare.values()].filter((v) => v.core > 0 && v.working > 0).length;
        const detail = `${coreRoles} core, ${workRoles} working${bothRoles ? `; ${bothRoles} with both` : ''}`;
        bar.innerHTML = `<span>Selected area: <strong>${esc(a?.code)} ${esc(a?.label)}</strong> — required by ${areaShare.size} of ${rolesOrdered.length} roles (${detail})</span>
          <button type="button" class="bok-viz-clear" data-clear-sankey="1">Clear selection</button>`;
      } else {
        bar.innerHTML = `<span>Select any <strong>role</strong> on the left or <strong>area</strong> on the right to isolate its footprint across the taxonomy.</span>`;
      }
    }
  }

  function drawSymptomFlow() {
    const stage = $('bok-symptom-stage');
    const detailEl = $('bok-symptom-detail');
    if (!stage) return;
    const g = data.framework.guides || { symptoms: [] };
    const symptoms = g.symptoms || [];
    if (!symptoms.length) return;
    const symId = state.activeSymptom || symptoms[0].id;
    state.activeSymptom = symId;
    const sym = symptoms.find((s) => s.id === symId) || symptoms[0];

    const selEl = $('bok-symptom-select');
    if (selEl && selEl.value !== sym.id) selEl.value = sym.id;
    for (const b of document.querySelectorAll('[data-symptom-pick]')) {
      b.setAttribute('aria-pressed', String(b.dataset.symptomPick === sym.id));
    }

    const kList = [...(sym.checks || [])];
    const cList = [...(sym.quantities || []), ...(sym.causes || [])];  // the measured quantity sits with the causes in the middle column
    const lList = [...(sym.levers || [])];
    const kSet = new Set(kList), cSet = new Set(cList), lSet = new Set(lList);
    const allSet = new Set([...kList, ...cList, ...lList]);

    const directEdges = (data.relationships || []).filter((r) => allSet.has(r.source) && allSet.has(r.target));

    // Order Causes by connectivity, then order Checks and Levers by barycenter against Causes.
    const cPos = new Map(cList.map((id, i) => [id, i]));
    const orderByCauses = (list) => [...list].sort((a, b) => {
      const bary = (id) => {
        const hits = directEdges.filter((r) => (r.source === id && cSet.has(r.target)) || (r.target === id && cSet.has(r.source)));
        if (!hits.length) return 99;
        return hits.reduce((acc, r) => acc + (cPos.get(cSet.has(r.source) ? r.source : r.target) ?? 0), 0) / hits.length;
      };
      return bary(a) - bary(b);
    });
    const orderedK = orderByCauses(kList);
    const orderedC = cList;
    const orderedL = orderByCauses(lList);

    const maxRows = Math.max(orderedK.length, orderedC.length, orderedL.length, 3);
    const rowStep = 46;
    const topPad = 46;
    const H = Math.max(250, topPad + maxRows * rowStep + 22);
    const W = 960;

    const colX = { K: 20, C: 366, L: 712 };
    const colW = 228;
    const nodeCoords = new Map();

    // Key by `${colKey}:${id}` so a concept listed in two columns renders in both without overwriting.
    const placeCol = (items, colKey) => {
      const startY = topPad + 14;
      items.forEach((id, i) => {
        nodeCoords.set(`${colKey}:${id}`, { id, x: colX[colKey], y: startY + i * rowStep, col: colKey, idx: i });
      });
    };
    placeCol(orderedK, 'K');
    placeCol(orderedC, 'C');
    placeCol(orderedL, 'L');

    // Resolve each relationship to the best (sourceCol, targetCol) endpoint pair.
    const resolveEndpoints = (rel) => {
      const uCols = ['K', 'C', 'L'].filter((c) => nodeCoords.has(`${c}:${rel.source}`));
      const vCols = ['K', 'C', 'L'].filter((c) => nodeCoords.has(`${c}:${rel.target}`));
      let best = null, bestScore = 99;
      for (const uc of uCols) {
        for (const vc of vCols) {
          // Prefer cross-column through C (score 1), then same-column (score 2), then K<->L (score 3).
          const score = (uc !== vc && (uc === 'C' || vc === 'C')) ? 1 : (uc === vc ? 2 : 3);
          if (score < bestScore) { bestScore = score; best = [nodeCoords.get(`${uc}:${rel.source}`), nodeCoords.get(`${vc}:${rel.target}`)]; }
        }
      }
      return best;
    };

    // Identify which nodes in Column C are directly measured by a Check (`measures` relation).
    const measuredByCheck = new Set();
    for (const r of directEdges) {
      if (r.relation === 'measures' && kSet.has(r.source) && cSet.has(r.target)) {
        measuredByCheck.add(r.target);
      }
    }

    // Determine active node and its truthful chain (direct edges + 1-hop onward across C to opposite outer column).
    const activeNode = state.activeDiagNode && allSet.has(state.activeDiagNode) ? state.activeDiagNode : null;
    const directSet = new Set();
    const viaSet = new Set();
    const viaMap = new Map();
    const activeNeighbors = new Set();

    if (activeNode) {
      activeNeighbors.add(activeNode);
      const isCauseOnly = cSet.has(activeNode) && !kSet.has(activeNode) && !lSet.has(activeNode);
      const touches = (r, id) => r.source === id || r.target === id;
      const other = (r, id) => (r.source === id ? r.target : r.source);
      const directList = directEdges.filter((r) => touches(r, activeNode));
      for (const r of directList) {
        directSet.add(r);
        activeNeighbors.add(r.source);
        activeNeighbors.add(r.target);
      }
      if (!isCauseOnly) {
        const wantLever = kSet.has(activeNode);
        const oppSet = wantLever ? lSet : kSet;
        for (const r1 of directList) {
          const mid = other(r1, activeNode);
          if (!cSet.has(mid)) continue;
          const onward = directEdges.filter((r2) => r2 !== r1 && !directSet.has(r2) && touches(r2, mid) && oppSet.has(other(r2, mid)));
          if (onward.length) {
            viaMap.set(mid, onward);
            for (const r2 of onward) {
              viaSet.add(r2);
              activeNeighbors.add(r2.source);
              activeNeighbors.add(r2.target);
            }
          }
        }
      }
    }

    // Compute per-port vertical offsets so multiple curves at the same box side fan out cleanly.
    const portSlots = new Map();
    for (const rel of directEdges) {
      const pts = resolveEndpoints(rel);
      if (!pts) continue;
      const [u, v] = pts;
      if (u.col === v.col) continue;
      const uSide = u.x < v.x ? 'R' : 'L';
      const vSide = u.x < v.x ? 'L' : 'R';
      push(portSlots, `${u.col}:${rel.source}__${uSide}`, { rel, otherY: v.y });
      push(portSlots, `${v.col}:${rel.target}__${vSide}`, { rel, otherY: u.y });
    }
    const portOffset = new Map();
    for (const [pk, list] of portSlots) {
      list.sort((a, b) => a.otherY - b.otherY);
      const step = Math.min(4.2, 24 / Math.max(1, list.length - 1));
      list.forEach((item, idx) => {
        const off = (idx - (list.length - 1) / 2) * step;
        portOffset.set(`${pk}__${item.rel.source}__${item.rel.target}__${item.rel.relation}`, off);
      });
    }

    let edgesHtml = '';
    let verbLabelsHtml = '';
    const placedVerbs = [];
    const activeVerbGroups = new Map();
    for (const rel of directEdges) {
      const pts = resolveEndpoints(rel);
      if (!pts) continue;
      const [u, v] = pts;
      const isDirect = directSet.has(rel);
      const isVia = viaSet.has(rel);
      const isEdgeActive = isDirect || isVia;

      const isLeverEdge = u.col === 'L' || v.col === 'L' || relInFamily(rel, 'levers');
      const sameCol = u.col === v.col;
      const skipMiddle = !sameCol && u.col !== 'C' && v.col !== 'C';
      let d = '';
      let vx = 0, vy = 0;
      if (sameCol) {
        const sideX = u.col === 'L' ? u.x + colW + 22 : u.x - 22;
        const anchorX = u.col === 'L' ? u.x + colW : u.x;
        vx = sideX;
        vy = (u.y + v.y) / 2;
        d = `M${anchorX},${u.y} Q${sideX},${vy} ${anchorX},${v.y}`;
      } else if (skipMiddle) {
        const leftFirst = u.x < v.x;
        const x1 = leftFirst ? u.x + colW : u.x;
        const x2 = leftFirst ? v.x : v.x + colW;
        const xm = (x1 + x2) / 2;
        const apexY = 34;
        const ym = 2 * apexY - 0.5 * (u.y + v.y);
        vx = xm;
        vy = apexY;
        d = `M${x1.toFixed(1)},${u.y.toFixed(1)} Q${xm.toFixed(1)},${ym.toFixed(1)} ${x2.toFixed(1)},${v.y.toFixed(1)}`;
      } else {
        const leftFirst = u.x < v.x;
        const uSide = leftFirst ? 'R' : 'L';
        const vSide = leftFirst ? 'L' : 'R';
        const uOff = portOffset.get(`${u.col}:${rel.source}__${uSide}__${rel.source}__${rel.target}__${rel.relation}`) || 0;
        const vOff = portOffset.get(`${v.col}:${rel.target}__${vSide}__${rel.source}__${rel.target}__${rel.relation}`) || 0;
        const x1 = leftFirst ? u.x + colW : u.x;
        const y1 = u.y + uOff;
        const x2 = leftFirst ? v.x : v.x + colW;
        const y2 = v.y + vOff;
        const mx = (x1 + x2) / 2;
        d = `M${x1.toFixed(1)},${y1.toFixed(1)} C${mx.toFixed(1)},${y1.toFixed(1)} ${mx.toFixed(1)},${y2.toFixed(1)} ${x2.toFixed(1)},${y2.toFixed(1)}`;
        vx = mx;
        vy = (y1 + y2) / 2 - 2;
      }
      const cls = [
        'viz-diag-edge',
        isLeverEdge ? 'is-lever' : '',
        sameCol ? 'is-causal' : '',
        activeNode ? (isDirect ? 'is-active' : (isVia ? 'is-active is-via' : 'is-dim')) : '',
      ].filter(Boolean).join(' ');
      const markerUrl = isEdgeActive ? 'url(#bok-diag-arrow-on)' : 'url(#bok-diag-arrow)';
      const vStr = verb(rel);
      edgesHtml += `<path class="${cls}" d="${d}" marker-end="${markerUrl}"><title>${esc(concepts.get(rel.source)?.label)} ${esc(vStr)} ${esc(concepts.get(rel.target)?.label)}</title></path>`;
      if (isEdgeActive && !sameCol) {
        const hubId = cSet.has(rel.source) ? rel.source : rel.target;
        const gk = `${u.col}__${v.col}__${hubId}__${vStr}`;
        push(activeVerbGroups, gk, { vx, vy, vStr });
      }
    }
    for (const grp of activeVerbGroups.values()) {
      grp.sort((a, b) => a.vy - b.vy);
      const mid = grp[Math.floor(grp.length / 2)];
      const labelStr = grp.length > 1 ? `${mid.vStr} (×${grp.length})` : mid.vStr;
      const boxW = Math.max(52, labelStr.length * 5.3 + 10);
      let vx = mid.vx, vy = mid.vy;
      while (placedVerbs.some((p) => Math.abs(p.x - vx) < (p.w + boxW) / 2 + 6 && Math.abs(p.y - vy) < 14)) {
        vy += 14;
      }
      placedVerbs.push({ x: vx, y: vy, w: boxW });
      verbLabelsHtml += `<g transform="translate(${vx.toFixed(1)},${vy.toFixed(1)})">
        <rect x="${(-boxW / 2).toFixed(1)}" y="-9" width="${boxW.toFixed(1)}" height="12" rx="2" fill="var(--bok-surface)" fill-opacity="0.94"/>
        <text x="0" y="0" text-anchor="middle" font-size="9" font-style="italic" fill="var(--bok-accent)" font-weight="600">${esc(labelStr)}</text>
      </g>`;
    }

    let nodesHtml = `
      <text x="${colX.K}" y="18" class="bok-svg-group-label">1. What to check (${orderedK.length})</text>
      <text x="${colX.C}" y="18" class="bok-svg-group-label">2. Symptom metric &amp; likely causes (${orderedC.length})</text>
      <text x="${colX.L}" y="18" class="bok-svg-group-label">3. What to change (${orderedL.length})</text>
      <g transform="translate(${colX.L - 36}, 33)" aria-hidden="true">
        <line x1="0" y1="-3" x2="15" y2="-3" stroke="var(--bok-ink)" stroke-width="1.4"/>
        <text x="19" y="0" class="bok-svg-label" style="font-size:8.5px;fill:var(--bok-faint)">measures / limits</text>
        <line x1="98" y1="-3" x2="113" y2="-3" stroke="var(--bok-ink)" stroke-width="1.4" stroke-dasharray="4 2.5"/>
        <text x="117" y="0" class="bok-svg-label" style="font-size:8.5px;fill:var(--bok-faint)">reduces / mitigates</text>
        <line x1="204" y1="-3" x2="219" y2="-3" stroke="var(--bok-ink)" stroke-width="1.4" stroke-dasharray="1.5 2"/>
        <text x="223" y="0" class="bok-svg-label" style="font-size:8.5px;fill:var(--bok-faint)">same-stage</text>
      </g>`;

    for (const [, pt] of nodeCoords) {
      const id = pt.id;
      const c = concepts.get(id);
      const r = homeOf.get(id);
      const lbl = c?.label || id;
      const shortLbl = lbl.length > 30 ? lbl.slice(0, 29) + '…' : lbl;
      const subjShort = r?.display_label || r?.label || '';
      const metricTag = pt.col === 'C' && measuredByCheck.has(id) ? ' · Metric' : '';
      const isAct = activeNode === id;
      const isNbr = activeNeighbors.has(id);
      const isDim = activeNode && !isNbr;
      const cls = ['viz-diag-node', isAct || isNbr ? 'is-active' : '', isDim ? 'is-dim' : ''].filter(Boolean).join(' ');
      nodesHtml += `<g class="${cls}" data-diag-node="${esc(id)}" transform="translate(${pt.x},${pt.y - 16})" role="button" tabindex="0" aria-label="${esc(lbl)}">
        <rect x="0" y="0" width="${colW}" height="32" rx="2" fill="var(--bok-surface)" stroke="${isAct ? 'var(--bok-accent)' : 'var(--bok-line)'}" stroke-width="${isAct ? '1.5' : '1'}"/>
        <rect class="viz-diag-tick" x="0" y="0" width="3" height="32" fill="${isAct || isNbr ? 'var(--bok-accent)' : 'var(--bok-ink)'}"/>
        <text x="10" y="12" class="bok-svg-mono" style="font-size:8.5px;fill:var(--bok-faint)">${esc(r?.code || '')} · ${esc(subjShort)}${esc(metricTag)}</text>
        <text class="viz-diag-title bok-svg-label" x="10" y="25" style="font-size:10.5px;font-weight:${isAct ? '700' : '500'}">${esc(shortLbl)}</text>
      </g>`;
    }

    stage.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="group" aria-labelledby="diag-title diag-desc">
      <title id="diag-title">Diagnostic triage graph for ${esc(sym.label)}</title>
      <desc id="diag-desc">Connects ${orderedK.length} diagnostic checks, ${orderedC.length} likely causes, and ${orderedL.length} mitigation levers via ${directEdges.length} graph propositions.</desc>
      <defs>
        <marker id="bok-diag-arrow" viewBox="0 0 8 8" refX="7.5" refY="4" markerWidth="6.5" markerHeight="6.5" markerUnits="userSpaceOnUse" orient="auto">
          <path d="M0,1.2 L7.5,4 L0,6.8 Z" fill="var(--bok-ink)" fill-opacity="0.65"/>
        </marker>
        <marker id="bok-diag-arrow-on" viewBox="0 0 8 8" refX="7.5" refY="4" markerWidth="6.5" markerHeight="6.5" markerUnits="userSpaceOnUse" orient="auto">
          <path d="M0,1.2 L7.5,4 L0,6.8 Z" fill="var(--bok-accent)"/>
        </marker>
      </defs>
      <g>${edgesHtml}</g>
      <g>${nodesHtml}</g>
      <g>${verbLabelsHtml}</g>
    </svg>`;

    if (detailEl) {
      const renderRelItem = (rel) => `<li style="break-inside:avoid">${conceptLink(rel.source)} <span class="bok-verb">${esc(verb(rel))}</span> ${conceptLink(rel.target)}${rel.conditions ? `<p class="bok-cond">${esc(rel.conditions)}</p>` : ''}</li>`;
      if (!activeNode) {
        detailEl.innerHTML = `<div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;padding-bottom:6px;border-bottom:1px solid var(--bok-line);font-size:0.82rem">
          <span><strong>${esc(sym.label)}:</strong> ${esc(sym.signal)} (${directEdges.length} graph propositions connecting checks, causes, and levers — select any box above to trace its chain)</span>
        </div>
        <ul class="bok-viz-prop-list" style="columns:2 320px;column-gap:24px;margin-top:6px !important">${directEdges.map(renderRelItem).join('')}</ul>`;
      } else {
        const directArr = [...directSet];
        const viaGroupsHtml = [...viaMap.entries()].map(([midId, list]) => `<div class="bok-viz-prop-group">
          <p class="bok-viz-prop-label">Connected via ${esc(concepts.get(midId)?.label || midId)} (${list.length})</p>
          <ul class="bok-viz-prop-list" style="columns:2 320px;column-gap:24px">${list.map(renderRelItem).join('')}</ul>
        </div>`).join('');
        detailEl.innerHTML = `<div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;padding-bottom:6px;border-bottom:1px solid var(--bok-line);font-size:0.82rem">
          <span>Showing chain for <strong>${conceptLink(activeNode)}</strong> (${directArr.length} direct${viaSet.size ? ` · ${viaSet.size} via connected cause` : ''})</span>
          <button type="button" class="bok-viz-clear" data-clear-diag="1">Show all ${directEdges.length} propositions</button>
        </div>
        <div class="bok-viz-prop-group">
          <p class="bok-viz-prop-label"><span class="bok-viz-swatch-out"></span>Direct propositions (${directArr.length})</p>
          ${directArr.length ? `<ul class="bok-viz-prop-list" style="columns:2 320px;column-gap:24px">${directArr.map(renderRelItem).join('')}</ul>` : '<p class="bok-muted">No direct edges within this symptom subset — open the concept page for its full neighborhood.</p>'}
        </div>
        ${viaGroupsHtml}`;
      }
    }
  }

  function buildPathArcHtml(lp) {
    const steps = lp.steps || [];
    const n = steps.length;
    if (!n) return '';
    const stepIndex = new Map(steps.map((st, i) => [st.subject, i]));
    const pairMap = new Map();

    for (const { rel, iu, iv } of crossStepRelationships(stepIndex)) {
      const key = `${iu}__${iv}`;
      if (!pairMap.has(key)) pairMap.set(key, { uIdx: iu, vIdx: iv, count: 0, rels: [] });
      const entry = pairMap.get(key);
      entry.count++;
      entry.rels.push(rel);
    }

    const W = 960, H = 196;
    const padX = 72;
    const baseY = 118;
    const stepX = (i) => padX + (n > 1 ? (i / (n - 1)) * (W - padX * 2) : (W - padX * 2) / 2);

    let arcsHtml = '';
    const totalProps = [...pairMap.values()].reduce((s, x) => s + x.count, 0);
    for (const arc of pairMap.values()) {
      const a = Math.min(arc.uIdx, arc.vIdx);
      const b = Math.max(arc.uIdx, arc.vIdx);
      const isFwd = arc.uIdx < arc.vIdx;
      const x1 = stepX(arc.uIdx), x2 = stepX(arc.vIdx);
      const span = b - a;
      const archBase = 18 + (span / Math.max(1, n - 1)) * 86;
      const archH = isFwd ? archBase : archBase * 0.82;
      const xm = (x1 + x2) / 2;
      const ym = baseY - archH * 1.65;
      const w = Math.min(2.3, 0.75 + Math.sqrt(arc.count) * 0.45).toFixed(2);
      arcsHtml += `<path class="viz-path-arc" d="M${x1.toFixed(1)},${baseY - 7} Q${xm.toFixed(1)},${ym.toFixed(1)} ${x2.toFixed(1)},${baseY - 7}" stroke-width="${w}" data-arc-u="${arc.uIdx}" data-arc-v="${arc.vIdx}" data-arc-count="${arc.count}"/>`;
    }

    let stationsHtml = `<line x1="${padX}" y1="${baseY}" x2="${W - padX}" y2="${baseY}" stroke="var(--bok-ink)" stroke-width="1.5"/>`;
    for (let i = 0; i < n; i++) {
      const st = steps[i];
      const r = rows.get(st.subject);
      const x = stepX(i);
      const stagger = i % 2 === 1;
      const lblY = stagger ? baseY + 42 : baseY + 22;
      const shortName = r?.display_label || r?.label || st.subject;
      stationsHtml += `<g class="viz-path-station" data-path-id="${esc(lp.id)}" data-station-idx="${i}" role="button" tabindex="0" aria-label="Step ${i + 1}: ${esc(r?.code || '')} ${esc(r?.label || st.subject)}">
        <line x1="${x.toFixed(1)}" y1="${baseY + 7}" x2="${x.toFixed(1)}" y2="${lblY - 9}" stroke="var(--bok-line-strong)" stroke-width="0.8"/>
        <circle cx="${x.toFixed(1)}" cy="${baseY}" r="7"/>
        <text x="${x.toFixed(1)}" y="${baseY + 3}" text-anchor="middle" class="viz-station-num bok-svg-mono" style="font-size:8px;font-weight:700">${i + 1}</text>
        <text x="${x.toFixed(1)}" y="${lblY}" text-anchor="middle" class="viz-station-code bok-svg-mono" style="font-size:9px;font-weight:700">${esc(r?.code || '')}</text>
        <text x="${x.toFixed(1)}" y="${lblY + 11}" text-anchor="middle" class="bok-svg-label" style="font-size:8.5px;fill:var(--bok-muted)">${esc(shortName)}</text>
      </g>`;
    }

    const defaultSummary = `<strong>Figure.</strong> ${n} ordered subjects connected by ${pairMap.size} directed cross-step arcs (${totalProps} propositions). Hover or select a station to inspect its outgoing (crimson) and incoming (ink) relationships along the path.`;
    return `<div class="bok-path-viz" data-path-viz="${esc(lp.id)}" data-default-summary="${esc(defaultSummary)}">
      <svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="group" aria-label="Cross-step relationship arc diagram for ${esc(lp.label)}">
        <g class="viz-layer-arcs">${arcsHtml}</g>
        <g class="viz-layer-stations">${stationsHtml}</g>
      </svg>
      <div class="bok-path-viz-status" id="path-status-${esc(lp.id)}">
        <span>${defaultSummary}</span>
      </div>
    </div>`;
  }

  // Guides: ways in for a role, a production symptom, or a learning goal. Every
  // entry links to the subject and concept pages; nothing here is a new claim.
  function renderGuides() {
    const g = data.framework.guides || { roles: [], symptoms: [], learning_paths: [] };
    if (!state.activeSymptom && g.symptoms?.length) state.activeSymptom = g.symptoms[0].id;
    const subj = (id) => (rows.has(id) ? `<a class="bok-guide-subject" href="${subjectHref(rows.get(id))}">${code(rows.get(id).code)}${esc(rows.get(id).display_label || rows.get(id).label)}</a>` : esc(id));
    const list = (ids, f) => (ids && ids.length ? `<ul class="bok-guide-list">${ids.map((id) => `<li>${f(id)}</li>`).join('')}</ul>` : '<span class="bok-muted">—</span>');
    const roles = g.roles.map((r) => `<tr id="guide-role-${esc(r.id)}"><td><strong>${esc(r.label)}</strong><span class="bok-alias">${esc(r.summary)}</span>
        <span style="display:block;margin-top:4px;font-size:0.76rem"><button type="button" class="bok-viz-clear" data-pick-role="${esc(r.id)}" style="margin-right:10px">Show in diagram &uarr;</button><a href="#overlay=role:${esc(r.id)}" class="bok-viz-clear">Highlight on Matrix &nearr;</a></span></td>
        <td>${list(r.core, subj)}</td><td>${list(r.working, subj)}</td></tr>`).join('');
    const symptoms = g.symptoms.map((s) => `<tr id="guide-symptom-${esc(s.id)}"><td><strong>${esc(s.label)}</strong><span class="bok-alias">${esc(s.signal)}</span>
        <span style="display:block;margin-top:4px;font-size:0.76rem"><button type="button" class="bok-viz-clear" data-symptom-pick="${esc(s.id)}">Open in triage diagram &uarr;</button></span>
        ${s.quantities && s.quantities.length ? `<span class="bok-alias" style="display:block;margin-top:4px">Measured by: ${s.quantities.map((id) => conceptLink(id)).join(', ')}</span>` : ''}</td>
        <td>${list(s.causes, conceptLink)}</td><td>${list(s.checks, conceptLink)}</td><td>${list(s.levers, conceptLink)}</td></tr>`).join('');
    const paths = g.learning_paths.map((lp) => `<section class="bok-guide-path" id="guide-path-${esc(lp.id)}">
        <div style="display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:8px">
          <h4>${esc(lp.label)}</h4>
          <a href="#overlay=path:${esc(lp.id)}" class="bok-viz-clear">Highlight path on Matrix &nearr;</a>
        </div>
        <p class="bok-lead">${esc(lp.audience)}</p>
        ${buildPathArcHtml(lp)}
        <ol>${lp.steps.map((st, idx) => `<li data-path-step-item="${esc(lp.id)}:${idx}">${subj(st.subject)} <span class="bok-muted">${esc(st.why)}</span></li>`).join('')}</ol></section>`).join('');

    const SYMPTOM_SHORT_LABELS = {
      'ttft-spike': 'TTFT under load',
      'training-oom': 'Training OOM',
      'low-utilization': 'Low GPU utilization',
      'loss-spikes': 'Loss spikes / NaNs',
      'slow-decode': 'Slow token decode',
      'quality-drop': 'Quality drop',
      'runaway-agent': 'Runaway agent',
    };
    const quickSymptomTabs = g.symptoms.slice(0, 7).map((s) => [
      s.id,
      SYMPTOM_SHORT_LABELS[s.id] || (s.label.length > 22 ? s.label.slice(0, 20) + '…' : s.label),
    ]);

    $('bok-guides').innerHTML = `<div class="bok-entry-head"><div><h2>Guides</h2><p class="bok-lead">Ways into the taxonomy for a role, a production symptom, or a learning goal. Each entry links to the subject and concept pages.</p></div></div>
      <nav class="bok-jump" aria-label="Jump to guide"><a href="#view=guides&amp;section=roles">By role</a><a href="#view=guides&amp;section=symptoms">By symptom</a><a href="#view=guides&amp;section=paths">Learning paths</a></nav>
      <h3 id="guide-section-roles">By role</h3><p class="bok-lead">${esc(g.roles_note || '')}</p>
      <figure class="bok-viz-figure" id="bok-role-sankey-fig">
        <div class="bok-viz-bar">
          <div id="bok-role-sankey-status" style="display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;font-size:0.82rem;color:var(--bok-muted)"></div>
          ${expandBtn('bok-role-sankey-fig')}
        </div>
        <div class="bok-viz-stage" id="bok-role-sankey-stage"></div>
        <figcaption class="bok-viz-caption"><strong>Figure 2. Role → group → area alluvial distribution.</strong> Core subjects count double and are drawn darker; working-knowledge subjects are lighter. Roles are ordered by vertical barycenter to minimize crossings; areas follow canonical matrix order.</figcaption>
      </figure>
      <table class="bok-table bok-guide-table"><thead><tr><th>Role</th><th>Core subjects</th><th>Working knowledge</th></tr></thead><tbody>${roles}</tbody></table>
      <h3 id="guide-section-symptoms">By symptom</h3><p class="bok-lead">${esc(g.symptoms_note || '')}</p>
      <figure class="bok-viz-figure" id="bok-symptom-flow-fig">
        <div class="bok-viz-bar">
          <div class="bok-viz-controls">
            <label for="bok-symptom-select" class="bok-filter-label">Symptom</label>
            <select id="bok-symptom-select" style="height:28px;padding:0 10px;border:1px solid var(--bok-line-strong);border-radius:4px;background:var(--bok-surface);color:var(--bok-ink);font:inherit;font-size:0.82rem">
              ${g.symptoms.map((s) => `<option value="${esc(s.id)}">${esc(s.label)}</option>`).join('')}
            </select>
          </div>
          <div class="bok-viz-seg" role="group" aria-label="Common production symptoms">
            ${quickSymptomTabs.map(([id, tabLbl]) => `<button type="button" data-symptom-pick="${esc(id)}" aria-pressed="${String(state.activeSymptom === id)}">${esc(tabLbl)}</button>`).join('')}
          </div>
          ${expandBtn('bok-symptom-flow-fig')}
        </div>
        <div class="bok-viz-body">
          <div class="bok-viz-stage" id="bok-symptom-stage"></div>
          <div id="bok-symptom-detail" class="bok-viz-panel"></div>
        </div>
        <figcaption class="bok-viz-caption"><strong>Figure 3. Three-stage diagnostic triage flow (What to check → Symptom metric &amp; likely causes ← What to change).</strong> Solid curves represent diagnostic measurement, limit, or causal relationships; dashed curves represent engineering mitigations and levers; dotted curves link concepts within the same stage. Select any concept box to isolate its action chain.</figcaption>
      </figure>
      <table class="bok-table bok-guide-table"><thead><tr><th>Symptom</th><th>Likely causes</th><th>Check</th><th>Change</th></tr></thead><tbody>${symptoms}</tbody></table>
      <h3 id="guide-section-paths">Learning paths</h3><p class="bok-lead">${esc(g.paths_note || '')}</p>${paths}`;

    drawRoleSankey();
    drawSymptomFlow();
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
    state.vizFull = null;
    const rawHash = location.hash.slice(1);
    if (rawHash === 'bok-how-to-use') {
      state.view = 'landing';
      render();
      const howTo = $('bok-how-to-use');
      if (howTo) {
        howTo.open = true;
        howTo.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
      return;
    }
    if (rawHash === 'bok-release') {
      if (!['landing', 'home'].includes(state.view)) state.view = 'landing';
      render();
      $('bok-release')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }
    const p = new URLSearchParams(rawHash);
    const view = p.get('view');
    const replaced = (data.retired || []).find((r) => r.id === p.get('concept'))?.replaced_by;
    const concept = concepts.has(p.get('concept')) ? p.get('concept') : replaced || null;
    const subject = [p.get('subject'), p.get('row')].find((id) => rows.has(id)) || null;
    const area = areas.some((a) => a.id === p.get('area')) ? p.get('area') : null;
    Object.assign(state, { area, subject, concept, to: concepts.has(p.get('to')) ? p.get('to') : null, term: concepts.has(p.get('term')) ? p.get('term') : null });
    if (p.get('overlay')) {
      state.overlay = p.get('overlay');
      state.pathStep = 0;
    } else if (location.hash && view !== 'matrix') {
      state.overlay = null;
      state.pathStep = 0;
    }
    if (p.get('role')) state.activeRole = p.get('role');
    if (p.get('symptom')) {
      state.activeSymptom = p.get('symptom');
      state.activeDiagNode = null;
    }
    const guide = ['role', 'symptom', 'path', 'section'].map((k) => p.get(k) && `${k}-${p.get(k)}`).find(Boolean) || null;
    state.guide = guide;
    if (view === 'guides') state.view = 'guides';
    else if (view === 'glossary' || view === 'index') state.view = 'glossary';
    else if (concept) state.view = 'concept';
    else if (view === 'matrix' || p.get('overlay')) state.view = 'home';
    else if (view === 'concept' || view === 'graph' || view === 'map') state.view = 'landing';
    else if (subject) state.view = 'subject';
    else if (area) state.view = 'area';
    else state.view = 'landing';
    render();
  }

  function markFull(fig, on) {
    fig.classList.toggle('is-full', on);
    if (on) {
      fig.setAttribute('role', 'dialog');
      fig.setAttribute('aria-modal', 'true');
      fig.setAttribute('aria-label', fig.querySelector('figcaption strong')?.textContent || 'Expanded figure');
    } else {
      ['role', 'aria-modal', 'aria-label'].forEach((a) => fig.removeAttribute(a));
    }
    const btn = fig.querySelector('[data-viz-expand]');
    if (!btn) return;
    btn.setAttribute('aria-pressed', String(on));
    btn.querySelector('path')?.setAttribute('d', on ? ICON_EXIT : ICON_EXPAND);
    const lbl = btn.querySelector('.bok-viz-expand-label');
    if (lbl) lbl.textContent = on ? 'Exit full screen (Esc)' : 'Expand to full screen';
  }

  function syncFull() {
    const fig = state.vizFull && document.getElementById(state.vizFull);
    const live = !!fig && !fig.closest('[hidden]');
    if (!live) state.vizFull = null;
    for (const f of document.querySelectorAll('.bok-viz-figure.is-full')) {
      if (f !== fig || !live) markFull(f, false);
    }
    if (live) markFull(fig, true);
    document.documentElement.classList.toggle('bok-viz-locked', live);
    for (const el of document.querySelectorAll('.bok > *')) {
      if (live && !el.contains(fig)) el.setAttribute('inert', '');
      else el.removeAttribute('inert');
    }
  }

  function clearVizSelection(figId) {
    const any = !figId;
    if ((any ? state.view === 'landing' : figId === 'bok-coupling-figure') && state.couplingSel) {
      state.couplingSel = null;
      refreshCouplingFigure();
      return true;
    }
    if ((any ? state.view === 'guides' : figId === 'bok-role-sankey-fig') && (state.activeRole || state.activeAreaRole)) {
      state.activeRole = null;
      state.activeAreaRole = null;
      drawRoleSankey();
      return true;
    }
    if ((any ? state.view === 'guides' : figId === 'bok-symptom-flow-fig') && state.activeDiagNode) {
      state.activeDiagNode = null;
      drawSymptomFlow();
      return true;
    }
    return false;
  }

  function render() {
    const v = state.view;
    const entry = ['area', 'subject', 'concept', 'landing'].includes(v);
    $('bok-home').hidden = v !== 'home';
    $('bok-entry').hidden = !entry;
    $('bok-tree').hidden = v === 'landing';
    $('bok-crumbs').hidden = v === 'landing';
    $('bok-entry').classList.toggle('is-full', v === 'landing');
    $('bok-glossary').hidden = v !== 'glossary';
    $('bok-guides').hidden = v !== 'guides';
    if ($('bok-release')) $('bok-release').hidden = !['landing', 'home'].includes(v);
    for (const [tab, views] of [['bok-tab-concept', ['concept', 'landing']], ['bok-tab-home', ['home', 'area', 'subject']], ['bok-tab-guides', ['guides']], ['bok-tab-glossary', ['glossary']]]) {
      if (views.includes(v)) $(tab).setAttribute('aria-current', 'page');
      else $(tab).removeAttribute('aria-current');
    }
    if (v === 'home') renderMatrix();
    else if (v === 'area') renderArea();
    else if (v === 'subject') renderSubject();
    else if (v === 'concept') renderConcept();
    else if (v === 'landing') renderConceptLanding();
    else if (v === 'glossary') filterGlossary();
    else if (v === 'guides') {
      if (!$('bok-guides').innerHTML) renderGuides();
      else { drawRoleSankey(); drawSymptomFlow(); }
    }
    syncFull();
    const target = v === 'glossary' && state.term ? document.getElementById(`term-${state.term}`)
      : v === 'guides' && state.guide ? document.getElementById(`guide-${state.guide}`) : null;
    if (target) {
      target.scrollIntoView({ block: 'center', behavior: 'smooth' });
      target.classList.add('is-focus');
      setTimeout(() => target.classList.remove('is-focus'), 1600);
    } else if (!state.vizFull) {
      const top = document.querySelector('.bok-toolbar');
      if (top) {
        const rect = top.getBoundingClientRect();
        if (rect.top < 0 || (location.hash && rect.top > window.innerHeight * 0.55)) {
          top.scrollIntoView({ block: 'start', behavior: 'smooth' });
        }
      }
    }
  }

  function printOutline() {
    document.body.classList.add('bok-printing');
    window.print();
    document.body.classList.remove('bok-printing');
  }

  function highlightPathStation(pathId, stationIdx) {
    const container = document.querySelector(`[data-path-viz="${pathId}"]`);
    if (!container) return;
    const g = data.framework.guides || { learning_paths: [] };
    const lp = (g.learning_paths || []).find((p) => p.id === pathId);
    if (!lp) return;

    for (const stEl of container.querySelectorAll('.viz-path-station')) {
      stEl.classList.toggle('is-active', Number(stEl.dataset.stationIdx) === stationIdx);
    }
    const stepIndex = new Map((lp.steps || []).map((st, i) => [st.subject, i]));
    let earlToThis = 0, thisToEarl = 0, thisToLate = 0, lateToThis = 0;
    const earlSet = new Set(), lateSet = new Set();
    const stationProps = [];

    for (const arcEl of container.querySelectorAll('.viz-path-arc')) {
      const u = Number(arcEl.dataset.arcU), v = Number(arcEl.dataset.arcV);
      arcEl.classList.remove('is-in', 'is-out', 'is-dim');
      if (stationIdx === null) continue;
      if (u === stationIdx) {
        arcEl.classList.add('is-out');
      } else if (v === stationIdx) {
        arcEl.classList.add('is-in');
      } else {
        arcEl.classList.add('is-dim');
      }
    }

    if (stationIdx !== null) {
      for (const { rel, iu, iv } of crossStepRelationships(stepIndex)) {
        if (iu !== stationIdx && iv !== stationIdx) continue;
        stationProps.push(rel);
        const otherIdx = iu === stationIdx ? iv : iu;
        const otherCode = rows.get(lp.steps[otherIdx]?.subject)?.code || `Step ${otherIdx + 1}`;
        if (otherIdx < stationIdx) {
          earlSet.add(otherCode);
          if (iu < stationIdx) earlToThis++;
          else thisToEarl++;
        } else {
          lateSet.add(otherCode);
          if (iu === stationIdx) thisToLate++;
          else lateToThis++;
        }
      }
    }

    const pathSec = container.closest('.bok-guide-path');
    if (pathSec) {
      for (const li of pathSec.querySelectorAll('[data-path-step-item]')) {
        li.classList.toggle('is-active', stationIdx !== null && li.dataset.pathStepItem === `${pathId}:${stationIdx}`);
      }
    }

    const statusEl = $(`path-status-${pathId}`);
    if (statusEl) {
      if (stationIdx === null) {
        statusEl.innerHTML = `<span>${container.dataset.defaultSummary || ''}</span>`;
        return;
      }
      const st = lp.steps[stationIdx];
      const r = rows.get(st?.subject);
      const earlTot = earlToThis + thisToEarl;
      const lateTot = thisToLate + lateToThis;
      const earlStr = earlTot
        ? `earlier steps (${[...earlSet].join(', ')}): ${earlTot} (${thisToEarl} outgoing · ${earlToThis} incoming)`
        : 'earlier steps: 0';
      const lateStr = lateTot
        ? `later steps (${[...lateSet].join(', ')}): ${lateTot} (${thisToLate} outgoing · ${lateToThis} incoming)`
        : 'later steps: 0';
      const propItems = stationProps.map((rel) => {
        const uS = homeOf.get(rel.source), vS = homeOf.get(rel.target);
        return `<li>${conceptLink(rel.source)} <span class="bok-mono" style="font-size:0.72rem">(${esc(uS?.code || '')})</span> <span class="bok-verb">${esc(verb(rel))}</span> ${conceptLink(rel.target)} <span class="bok-mono" style="font-size:0.72rem">(${esc(vS?.code || '')})</span></li>`;
      }).join('');
      statusEl.innerHTML = `<div style="width:100%">
        <div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px">
          <span><strong>Step ${stationIdx + 1} (${esc(r?.code || '')} ${esc(r?.display_label || r?.label || '')}):</strong> ${esc(earlStr)} · ${esc(lateStr)}</span>
          <button type="button" class="bok-viz-clear" data-clear-station="${esc(pathId)}">Clear</button>
        </div>
        ${stationProps.length ? `<details class="bok-viz-more"><summary>Show the ${stationProps.length} cross-step propositions for Step ${stationIdx + 1}</summary><ul class="bok-viz-prop-list" style="columns:2 300px;column-gap:20px;margin-top:4px !important">${propItems}</ul></details>` : ''}
      </div>`;
    }
  }

  function bind() {
    window.addEventListener('hashchange', restore);
    document.querySelectorAll('[data-try-lens]').forEach((a) => a.addEventListener('click', (e) => {
      e.preventDefault();
      if (location.hash !== '#view=matrix') {
        history.pushState(null, '', location.pathname + location.search + '#view=matrix');
        restore();
      }
      const b = document.querySelector(`#bok-lenses [data-lens="${a.dataset.tryLens}"]`);
      if (b && b.getAttribute('aria-pressed') !== 'true') b.click();
      $('bok-matrix').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }));
    $('bok-tab-concept').addEventListener('click', (e) => { e.preventDefault(); history.pushState(null, '', location.pathname + location.search); restore(); });
    $('bok-lenses').addEventListener('click', (e) => {
      const button = e.target.closest('button[data-lens]');
      if (!button) return;
      state.lens = state.lens === button.dataset.lens ? null : button.dataset.lens;
      renderMatrix();
    });
    $('bok-overlays')?.addEventListener('change', (e) => {
      if (e.target.id !== 'bok-overlay-select') return;
      state.overlay = e.target.value || null;
      state.pathStep = 0;
      renderMatrix();
    });
    $('bok-overlay-banner')?.addEventListener('click', (e) => {
      if (e.target.closest('[data-clear-overlay]')) {
        state.overlay = null;
        state.pathStep = 0;
        if (location.hash.startsWith('#overlay=')) history.pushState(null, '', location.pathname + location.search + '#view=matrix');
        renderMatrix();
        return;
      }
      const stepBtn = e.target.closest('[data-path-step]');
      if (stepBtn) {
        state.pathStep = Number(stepBtn.dataset.pathStep);
        renderMatrix();
        return;
      }
      const deltaBtn = e.target.closest('[data-step-delta]');
      if (deltaBtn) {
        state.pathStep = Math.max(0, state.pathStep + Number(deltaBtn.dataset.stepDelta));
        renderMatrix();
      }
    });
    $('bok-show-concepts').addEventListener('click', () => { state.showConcepts = !state.showConcepts; renderMatrix(); });
    $('bok-gloss-filter').addEventListener('input', filterGlossary);
    $('bok-gloss-area').addEventListener('input', filterGlossary);
    $('bok-print').addEventListener('click', (e) => { e.preventDefault(); printOutline(); });
    document.addEventListener('click', (e) => {
      if (e.target.closest('[data-print-outline]')) {
        e.preventDefault();
        printOutline();
        return;
      }
      const expandTrigger = e.target.closest('[data-viz-expand]');
      if (expandTrigger) {
        const figId = expandTrigger.dataset.vizExpand;
        state.vizFull = state.vizFull === figId ? null : figId;
        syncFull();
        expandTrigger.focus({ preventScroll: true });
        if (!state.vizFull) expandTrigger.closest('figure')?.scrollIntoView({ block: 'nearest' });
        return;
      }
      const letter = e.target.closest('a[data-letter]');
      if (letter) {
        e.preventDefault();
        document.getElementById(`letter-${letter.dataset.letter}`)?.scrollIntoView({ block: 'start', behavior: 'smooth' });
        return;
      }
      // Coupling figure interactions
      const modeBtn = e.target.closest('[data-coupling-mode]');
      if (modeBtn) {
        state.couplingMode = modeBtn.dataset.couplingMode;
        refreshCouplingFigure();
        return;
      }
      const famBtn = e.target.closest('[data-coupling-fam]');
      if (famBtn) {
        state.couplingFamily = famBtn.dataset.couplingFam;
        refreshCouplingFigure();
        return;
      }
      if (e.target.closest('[data-clear-coupling]')) {
        state.couplingSel = null;
        refreshCouplingFigure();
        return;
      }
      const pairBtn = e.target.closest('[data-select-pair]');
      if (pairBtn) {
        const [fromArea, toArea] = pairBtn.dataset.selectPair.split('__');
        state.couplingSel = { kind: 'pair', fromArea, toArea };
        refreshCouplingFigure();
        return;
      }
      const wheelSubj = e.target.closest('[data-wheel-subj]');
      if (wheelSubj) {
        const id = wheelSubj.dataset.wheelSubj;
        state.couplingSel = state.couplingSel?.kind === 'subject' && state.couplingSel.id === id ? null : { kind: 'subject', id };
        refreshCouplingFigure();
        return;
      }
      const areaTrigger = e.target.closest('[data-wheel-area], [data-dsm-area]');
      if (areaTrigger) {
        const id = areaTrigger.dataset.wheelArea || areaTrigger.dataset.dsmArea;
        state.couplingSel = state.couplingSel?.kind === 'area' && state.couplingSel.id === id ? null : { kind: 'area', id };
        refreshCouplingFigure();
        return;
      }
      const dsmCell = e.target.closest('[data-dsm-from]');
      if (dsmCell) {
        const fromArea = dsmCell.dataset.dsmFrom, toArea = dsmCell.dataset.dsmTo;
        state.couplingSel = state.couplingSel?.kind === 'pair' && state.couplingSel.fromArea === fromArea && state.couplingSel.toArea === toArea
          ? null : { kind: 'pair', fromArea, toArea };
        refreshCouplingFigure();
        return;
      }
      // Guides: Role Sankey interactions
      const sankeyRole = e.target.closest('[data-sankey-role]');
      if (sankeyRole) {
        const id = sankeyRole.dataset.sankeyRole;
        state.activeRole = state.activeRole === id ? null : id;
        state.activeAreaRole = null;
        drawRoleSankey();
        return;
      }
      const pickRole = e.target.closest('[data-pick-role]');
      if (pickRole) {
        state.activeRole = pickRole.dataset.pickRole;
        state.activeAreaRole = null;
        drawRoleSankey();
        $('bok-role-sankey-fig')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        return;
      }
      const sankeyArea = e.target.closest('[data-sankey-area]');
      if (sankeyArea && !e.target.closest('[data-sankey-role]')) {
        const id = sankeyArea.dataset.sankeyArea;
        state.activeAreaRole = state.activeAreaRole === id ? null : id;
        state.activeRole = null;
        drawRoleSankey();
        return;
      }
      if (e.target.closest('[data-clear-sankey]')) {
        state.activeRole = null;
        state.activeAreaRole = null;
        drawRoleSankey();
        return;
      }
      // Guides: Symptom triage flow interactions
      const symPick = e.target.closest('[data-symptom-pick]');
      if (symPick) {
        state.activeSymptom = symPick.dataset.symptomPick;
        state.activeDiagNode = null;
        drawSymptomFlow();
        if (symPick.closest('table')) $('bok-symptom-flow-fig')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        return;
      }
      const diagNode = e.target.closest('[data-diag-node]');
      if (diagNode) {
        const id = diagNode.dataset.diagNode;
        state.activeDiagNode = state.activeDiagNode === id ? null : id;
        drawSymptomFlow();
        return;
      }
      if (e.target.closest('[data-clear-diag]')) {
        state.activeDiagNode = null;
        drawSymptomFlow();
        return;
      }
      // Guides: Learning path station interactions
      const clearStation = e.target.closest('[data-clear-station]');
      if (clearStation) {
        const pid = clearStation.dataset.clearStation;
        const container = document.querySelector(`[data-path-viz="${pid}"]`);
        if (container) delete container.dataset.pinnedStation;
        highlightPathStation(pid, null);
        return;
      }
      const pathStation = e.target.closest('[data-station-idx]');
      if (pathStation) {
        const pid = pathStation.dataset.pathId;
        const idx = Number(pathStation.dataset.stationIdx);
        const container = document.querySelector(`[data-path-viz="${pid}"]`);
        if (container) {
          if (container.dataset.pinnedStation === String(idx)) {
            delete container.dataset.pinnedStation;
            highlightPathStation(pid, null);
          } else {
            container.dataset.pinnedStation = String(idx);
            highlightPathStation(pid, idx);
          }
        }
      }
    });
    document.addEventListener('change', (e) => {
      if (e.target.id === 'bok-symptom-select') {
        state.activeSymptom = e.target.value;
        state.activeDiagNode = null;
        drawSymptomFlow();
      }
    });
    document.addEventListener('mouseover', (e) => {
      const pathStation = e.target.closest('[data-station-idx]');
      if (pathStation) {
        const pid = pathStation.dataset.pathId;
        const container = document.querySelector(`[data-path-viz="${pid}"]`);
        if (container && container.dataset.pinnedStation !== undefined) return;
        highlightPathStation(pid, Number(pathStation.dataset.stationIdx));
      }
    });
    document.addEventListener('mouseout', (e) => {
      const pathStation = e.target.closest('[data-station-idx]');
      if (pathStation && !pathStation.contains(e.relatedTarget)) {
        const pid = pathStation.dataset.pathId;
        const container = document.querySelector(`[data-path-viz="${pid}"]`);
        if (container && container.dataset.pinnedStation === undefined) {
          highlightPathStation(pid, null);
        }
      }
    });
    wirePicker('bok-search', 'bok-results', choose);
    document.addEventListener('keydown', (e) => {
      if ((e.key === 'Enter' || e.key === ' ') && document.activeElement?.matches('[role="button"]:not(button)')) {
        e.preventDefault();
        document.activeElement.click();
        return;
      }
      if (e.key === '/' && !state.vizFull && !/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName)) {
        e.preventDefault();
        $('bok-search').focus();
        $('bok-search').select();
      } else if (e.key === 'Escape') {
        if (clearVizSelection(state.vizFull)) return;
        if (state.vizFull) {
          const btn = document.querySelector(`[data-viz-expand="${state.vizFull}"]`);
          state.vizFull = null;
          syncFull();
          btn?.focus({ preventScroll: true });
        }
      } else if (state.view === 'home' && state.overlay?.startsWith('path:') && !/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName)) {
        if (e.key === 'ArrowRight') { state.pathStep++; renderMatrix(); }
        else if (e.key === 'ArrowLeft') { state.pathStep = Math.max(0, state.pathStep - 1); renderMatrix(); }
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
