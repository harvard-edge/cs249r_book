/* Quarto owns the figure/anchor/caption; this controller owns only its views. */
(() => {
  'use strict';
  const mounted = new WeakMap();
  let svgSerial = 0;
  const namespaceArtwork = (root, figureId) => {
    // Namespace verified source artwork before the factory captures IDs or
    // fragment references for subsequent animation frames.
    for (const svg of (root.matches?.('svg') ? [root] : root.querySelectorAll('svg'))) {
      const prefix = figureId + '-explore-svg-' + (++svgSerial) + '-';
      const ids = new Map();
      for (const element of [svg, ...svg.querySelectorAll('[id]')]) {
        if (!element.id) continue;
        if (ids.has(element.id)) throw new Error('Duplicate IDs inside lesson SVG.');
        ids.set(element.id, prefix + element.id);
      }
      const reference = value => value.replace(/url\(\s*(["']?)#([^\s)'"%]+)\1\s*\)/g,
        (whole, quote, id) => ids.has(id) ? 'url(' + quote + '#' + ids.get(id) + quote + ')' : whole);
      for (const element of [svg, ...svg.querySelectorAll('*')]) {
        for (const attribute of [...element.attributes]) {
          let value = reference(attribute.value);
          if (attribute.localName === 'href' && value.startsWith('#') && ids.has(value.slice(1))) value = '#' + ids.get(value.slice(1));
          if (['aria-labelledby','aria-describedby'].includes(attribute.name)) value = value.split(/\s+/).map(id => ids.get(id) || id).join(' ');
          if (value !== attribute.value) {
            if (attribute.namespaceURI) element.setAttributeNS(attribute.namespaceURI, attribute.name, value);
            else element.setAttribute(attribute.name, value);
          }
        }
        if (element.localName === 'style') {
          let css = reference(element.textContent);
          for (const [id, replacement] of ids) {
            const escaped = id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            css = css.replace(new RegExp('#' + escaped + '(?![\\w:.-])', 'g'), '#' + replacement);
          }
          element.textContent = css;
        }
      }
      for (const element of [svg, ...svg.querySelectorAll('[id]')]) if (ids.has(element.id)) element.id = ids.get(element.id);
    }
  };
  const mount = async outer => {
    if (mounted.has(outer) || outer.dataset.mlsysExploreState === 'failed') return;
    mounted.set(outer, null);
    const factory = window.MLSysFigureLessons?.[outer.dataset.mlsysExplore];
    const figure = outer.matches('figure') ? outer : outer.querySelector(':scope > figure');
    const caption = figure?.querySelector(':scope > figcaption');
    if (typeof factory !== 'function' || !figure || !caption || !window.MLSysFigureViews) {
      mounted.delete(outer);
      console.warn('[figure-explore] Incomplete lesson or figure; keeping Diagram.', outer.id);
      return;
    }
    const original = [...figure.childNodes];
    const diagram = document.createElement('div');
    const explore = document.createElement('div');
    diagram.id = outer.id + '-diagram';
    explore.id = outer.id + '-explore';
    explore.hidden = true;
    let controller;
    try {
      // Prepare off-document: a failed lesson must not disturb the static figure.
      const key = outer.dataset.mlsysExploreKey;
      const lesson = window.MLSysGuidedLessons?.[key];
      const verifiedArtwork = outer.dataset.mlsysArtSha1
        ? await verifyArtwork(outer, original.filter(node => node !== caption)) : null;
      if (!outer.isConnected || figure.parentElement !== outer && figure !== outer) { mounted.delete(outer); return; }
      if (verifiedArtwork) namespaceArtwork(verifiedArtwork, outer.id);
      controller = await factory(explore, {outer, figure, staticNodes:original.filter(node => node !== caption), lesson, verifiedArtwork});
      if (typeof controller?.pause !== 'function') throw new Error('Lesson must supply pause().');
      for (const node of original) if (node !== caption) diagram.append(node);
      figure.insertBefore(diagram, caption);
      figure.insertBefore(explore, caption);
      figure.addEventListener('mlsys-view-change', event => {
        if (event.detail.view !== 'explore') controller.pause();
      });
      const requestedExplore = new URLSearchParams(location.search).get('view') === 'explore' && location.hash === '#' + outer.id;
      window.MLSysFigureViews.enhance({figure, diagram, explore, initialView:requestedExplore ? 'explore' : 'diagram'});
      if (requestedExplore) requestAnimationFrame(() => {
        const header = document.querySelector('#quarto-header');
        outer.style.scrollMarginTop = `${Math.max(0, header?.getBoundingClientRect().bottom || 0) + 16}px`;
        outer.scrollIntoView({block:'start'});
      });
      mounted.set(outer, controller);
      outer.dataset.mlsysExploreState = 'ready';
    } catch (error) {
      controller?.destroy?.();
      figure.replaceChildren(...original);
      figure.classList.remove('mlsys-figure-views');
      delete figure.dataset.figureViewsReady;
      delete figure.dataset.figureView;
      mounted.delete(outer);
      outer.dataset.mlsysExploreState = 'failed';
      console.error('[figure-explore] Initialization failed; keeping Diagram.', error);
    }
  };
  // Freshness check, not a security boundary. Match the compiler's exact SVG normalization.
  const normalizeSvg = text => {
    const ids = new Map();
    text = text.replace(/<metadata\b[^>]*>[\s\S]*?<\/metadata\s*>/g, '');
    text = text.replace(/\bid=(["'])([^"']+)\1/g, (_, quote, id) => {
      if (ids.has(id)) throw new Error('Duplicate SVG IDs in canonical artwork.');
      const replacement = 'id' + ids.size; ids.set(id, replacement);
      return 'id=' + quote + replacement + quote;
    });
    return text.replace(/#([A-Za-z_][\w:.-]*)/g, (whole, id) => ids.has(id) ? '#' + ids.get(id) : whole);
  };
  const verifiedPng = async (buffer, outer) => {
    const bytes = new Uint8Array(buffer);
    const signature = [137,80,78,71,13,10,26,10];
    if (bytes.length < 33 || signature.some((value, index) => bytes[index] !== value)) {
      throw new Error('Invalid canonical PNG artwork.');
    }
    const header = new DataView(buffer);
    if (header.getUint32(8) !== 13 || String.fromCharCode(...bytes.subarray(12,16)) !== 'IHDR') {
      throw new Error('Invalid canonical PNG dimensions.');
    }
    const width = header.getUint32(16), height = header.getUint32(20);
    if (!width || !height || width > 0x7fffffff || height > 0x7fffffff) {
      throw new Error('Invalid canonical PNG dimensions.');
    }
    for (const [name, value] of [['mlsysArtWidth', width], ['mlsysArtHeight', height]]) {
      if (outer.dataset[name] !== undefined && Number(outer.dataset[name]) !== value) {
        throw new Error('PNG dimensions changed since region review.');
      }
    }
    // Reuse the verified bytes: a wrapper must never fetch an unchecked image.
    const chunks = [];
    for (let offset = 0; offset < bytes.length; offset += 0x8000) {
      chunks.push(String.fromCharCode(...bytes.subarray(offset, offset + 0x8000)));
    }
    const source = 'data:image/png;base64,' + btoa(chunks.join(''));
    const decoded = new Image();
    decoded.src = source;
    await decoded.decode();
    if (decoded.naturalWidth !== width || decoded.naturalHeight !== height) {
      throw new Error('Canonical PNG could not be decoded at its reviewed dimensions.');
    }
    const namespace = 'http://www.w3.org/2000/svg';
    const wrapper = document.createElementNS(namespace, 'svg');
    wrapper.setAttribute('viewBox', `0 0 ${width} ${height}`);
    wrapper.setAttribute('width', String(width));
    wrapper.setAttribute('height', String(height));
    wrapper.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    wrapper.dataset.mlsysArtworkKind = 'png';
    wrapper.dataset.mlsysArtworkWidth = String(width);
    wrapper.dataset.mlsysArtworkHeight = String(height);
    const image = document.createElementNS(namespace, 'image');
    image.setAttribute('x', '0');
    image.setAttribute('y', '0');
    image.setAttribute('width', String(width));
    image.setAttribute('height', String(height));
    image.setAttribute('href', source);
    wrapper.append(image);
    return wrapper;
  };
  const verifyArtwork = async (outer, nodes) => {
    const images = [];
    for (const node of nodes) if (node.nodeType === 1) {
      if (node.matches('img')) images.push(node);
      images.push(...node.querySelectorAll('img'));
    }
    if (images.length !== 1 || !crypto.subtle) throw new Error('Cannot verify canonical artwork.');
    const original = images[0], source = original.currentSrc || original.src;
    const response = await fetch(source, {credentials:'same-origin'});
    if (!response.ok) throw new Error('Canonical artwork could not be fetched.');
    let bytes = await response.arrayBuffer();
    const svgText = outer.dataset.mlsysArtKind === 'svg' ? new TextDecoder().decode(bytes) : null;
    if (svgText !== null) bytes = new TextEncoder().encode(normalizeSvg(svgText));
    const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-1', bytes))].map(n => n.toString(16).padStart(2,'0')).join('');
    if (hash !== outer.dataset.mlsysArtSha1 || source !== (original.currentSrc || original.src)) {
      throw new Error('Artwork changed since region review; review and recompile the lesson.');
    }
    if (svgText === null) {
      if (outer.dataset.mlsysArtKind !== 'png') return null;
      const wrapper = await verifiedPng(bytes, outer);
      if (source !== (original.currentSrc || original.src)) {
        throw new Error('Artwork changed while the canonical PNG was decoded.');
      }
      return wrapper;
    }
    const parsed = new DOMParser().parseFromString(svgText, 'image/svg+xml');
    if (parsed.querySelector('parsererror') || parsed.documentElement.localName !== 'svg') throw new Error('Invalid canonical SVG artwork.');
    return document.importNode(parsed.documentElement, true);
  };
  const initialize = (root = document) => {
    if (root.matches?.('[data-mlsys-explore]')) mount(root);
    root.querySelectorAll('[data-mlsys-explore]').forEach(mount);
  };
  window.MLSysFigureExplore = Object.freeze({initialize});
  if (document.readyState === 'complete') initialize();
  else document.addEventListener('DOMContentLoaded', () => initialize(), {once:true});
})();
