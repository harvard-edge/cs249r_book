/* Quarto owns the figure/anchor/caption; this controller owns only its views. */
(() => {
  'use strict';
  const mounted = new WeakMap();
  const mount = outer => {
    if (mounted.has(outer) || outer.dataset.mlsysExploreState === 'failed') return;
    const factory = window.MLSysFigureLessons?.[outer.dataset.mlsysExplore];
    const figure = outer.matches('figure') ? outer : outer.querySelector(':scope > figure');
    const caption = figure?.querySelector(':scope > figcaption');
    if (typeof factory !== 'function' || !figure || !caption || !window.MLSysFigureViews) {
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
      controller = factory(explore);
      if (typeof controller?.pause !== 'function') throw new Error('Lesson must supply pause().');
      for (const node of original) if (node !== caption) diagram.append(node);
      figure.insertBefore(diagram, caption);
      figure.insertBefore(explore, caption);
      figure.addEventListener('mlsys-view-change', event => {
        if (event.detail.view !== 'explore') controller.pause();
      });
      window.MLSysFigureViews.enhance({figure, diagram, explore, initialView:'diagram'});
      mounted.set(outer, controller);
      outer.dataset.mlsysExploreState = 'ready';
    } catch (error) {
      controller?.destroy?.();
      figure.replaceChildren(...original);
      figure.classList.remove('mlsys-figure-views');
      delete figure.dataset.figureViewsReady;
      delete figure.dataset.figureView;
      outer.dataset.mlsysExploreState = 'failed';
      console.error('[figure-explore] Initialization failed; keeping Diagram.', error);
    }
  };
  const initialize = (root = document) => {
    if (root.matches?.('[data-mlsys-explore]')) mount(root);
    root.querySelectorAll('[data-mlsys-explore]').forEach(mount);
  };
  window.MLSysFigureExplore = Object.freeze({initialize});
  if (document.readyState === 'complete') initialize();
  else document.addEventListener('DOMContentLoaded', () => initialize(), {once:true});
})();
