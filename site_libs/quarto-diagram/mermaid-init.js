// mermaid-init.js
// Initializes the quarto-mermaid JS runtime
//
// Copyright (C) 2022 Posit Software, PBC

/**
 * String.prototype.replaceAll() polyfill
 * https://gomakethings.com/how-to-replace-a-section-of-a-string-with-another-one-with-vanilla-js/
 * @author Chris Ferdinandi
 * @license MIT
 */
if (!String.prototype.replaceAll) {
  String.prototype.replaceAll = function (str, newStr) {
    // If a regex pattern
    if (
      Object.prototype.toString.call(str).toLowerCase() === "[object regexp]"
    ) {
      return this.replace(str, newStr);
    }

    // If a string
    return this.replace(new RegExp(str, "g"), newStr);
  };
}

const mermaidOpts = {
  startOnLoad: false,
};
// this CSS is adapted from
// mkdocs-material
// Copyright (c) 2016-2022 Martin Donath <martin.donath@squidfunk.com>

const defaultCSS =
  '.label text {fill: var(--mermaid-fg-color);}.node circle, .node ellipse, .node path, .node polygon, .node rect {fill: var(--mermaid-node-bg-color);stroke: var(--mermaid-node-fg-color);}marker {fill: var(--mermaid-edge-color) !important;}.edgeLabel .label rect {fill: #0000;}.label {color: var(--mermaid-label-fg-color);font-family: var(--mermaid-font-family);}.label foreignObject {line-height: normal;overflow: visible;}.label div .edgeLabel {color: var(--mermaid-label-fg-color);}.edgeLabel, .edgeLabel rect, .label div .edgeLabel {background-color: var(--mermaid-label-bg-color);}.edgeLabel, .edgeLabel rect {fill: var(--mermaid-label-bg-color);color: var(--mermaid-edge-color);}.edgePath .path, .flowchart-link {stroke: var(--mermaid-edge-color);}.edgePath .arrowheadPath {fill: var(--mermaid-edge-color);stroke: none;}.cluster rect {fill: var(--mermaid-fg-color--lightest);stroke: var(--mermaid-fg-color--lighter);}.cluster span {color: var(--mermaid-label-fg-color);font-family: var(--mermaid-font-family);}defs #flowchart-circleEnd, defs #flowchart-circleStart, defs #flowchart-crossEnd, defs #flowchart-crossStart, defs #flowchart-pointEnd, defs #flowchart-pointStart {stroke: none;}g.classGroup line, g.classGroup rect {fill: var(--mermaid-node-bg-color);stroke: var(--mermaid-node-fg-color);}g.classGroup text {fill: var(--mermaid-label-fg-color);font-family: var(--mermaid-font-family);}.classLabel .box {fill: var(--mermaid-label-bg-color);background-color: var(--mermaid-label-bg-color);opacity: 1;}.classLabel .label {fill: var(--mermaid-label-fg-color);font-family: var(--mermaid-font-family);}.node .divider {stroke: var(--mermaid-node-fg-color);}.relation {stroke: var(--mermaid-edge-color);}.cardinality {fill: var(--mermaid-label-fg-color);font-family: var(--mermaid-font-family);}.cardinality text {fill: inherit !important;}defs #classDiagram-compositionEnd, defs #classDiagram-compositionStart, defs #classDiagram-dependencyEnd, defs #classDiagram-dependencyStart, defs #classDiagram-extensionEnd, defs #classDiagram-extensionStart {fill: var(--mermaid-edge-color) !important;stroke: var(--mermaid-edge-color) !important;}defs #classDiagram-aggregationEnd, defs #classDiagram-aggregationStart {fill: var(--mermaid-label-bg-color) !important;stroke: var(--mermaid-edge-color) !important;}g.stateGroup rect {fill: var(--mermaid-node-bg-color);stroke: var(--mermaid-node-fg-color);}g.stateGroup .state-title {fill: var(--mermaid-label-fg-color) !important;font-family: var(--mermaid-font-family);}g.stateGroup .composit {fill: var(--mermaid-label-bg-color);}.nodeLabel {color: var(--mermaid-label-fg-color);font-family: var(--mermaid-font-family);}.node circle.state-end, .node circle.state-start, .start-state {fill: var(--mermaid-edge-color);stroke: none;}.end-state-inner, .end-state-outer {fill: var(--mermaid-edge-color);}.end-state-inner, .node circle.state-end {stroke: var(--mermaid-label-bg-color);}.transition {stroke: var(--mermaid-edge-color);}[id^="state-fork"] rect, [id^="state-join"] rect {fill: var(--mermaid-edge-color) !important;stroke: none !important;}.statediagram-cluster.statediagram-cluster .inner {fill: var(--mermaid-bg-color);}.statediagram-cluster rect {fill: var(--mermaid-node-bg-color);stroke: var(--mermaid-node-fg-color);}.statediagram-state rect.divider {fill: var(--mermaid-fg-color--lightest);stroke: var(--mermaid-fg-color--lighter);}defs #statediagram-barbEnd {stroke: var(--mermaid-edge-color);}.entityBox {fill: var(--mermaid-label-bg-color);stroke: var(--mermaid-node-fg-color);}.entityLabel {fill: var(--mermaid-label-fg-color);font-family: var(--mermaid-font-family);}.relationshipLabelBox {fill: var(--mermaid-label-bg-color);fill-opacity: 1;background-color: var(--mermaid-label-bg-color);opacity: 1;}.relationshipLabel {fill: var(--mermaid-label-fg-color);}.relationshipLine {stroke: var(--mermaid-edge-color);}defs #ONE_OR_MORE_END *, defs #ONE_OR_MORE_START *, defs #ONLY_ONE_END *, defs #ONLY_ONE_START *, defs #ZERO_OR_MORE_END *, defs #ZERO_OR_MORE_START *, defs #ZERO_OR_ONE_END *, defs #ZERO_OR_ONE_START * {stroke: var(--mermaid-edge-color) !important;}.actor, defs #ZERO_OR_MORE_END circle, defs #ZERO_OR_MORE_START circle {fill: var(--mermaid-label-bg-color);}.actor {stroke: var(--mermaid-node-fg-color);}text.actor > tspan {fill: var(--mermaid-label-fg-color);font-family: var(--mermaid-font-family);}line {stroke: var(--mermaid-fg-color--lighter);}.messageLine0, .messageLine1 {stroke: var(--mermaid-edge-color);}.loopText > tspan, .messageText, .noteText > tspan {fill: var(--mermaid-edge-color);stroke: none;font-family: var(--mermaid-font-family) !important;}.noteText > tspan {fill: #000;}#arrowhead path {fill: var(--mermaid-edge-color);stroke: none;}.loopLine {stroke: var(--mermaid-node-fg-color);}.labelBox, .loopLine {fill: var(--mermaid-node-bg-color);}.labelBox {stroke: none;}.labelText, .labelText > span {fill: var(--mermaid-node-fg-color);font-family: var(--mermaid-font-family);}';

const mermaidThemeEl = document.querySelector('meta[name="mermaid-theme"]');
if (mermaidThemeEl) {
  mermaidOpts.theme = mermaidThemeEl.content;
} else {
  mermaidOpts.themeCSS = defaultCSS;
}

mermaid.initialize(mermaidOpts);

const _quartoMermaid = {
  // NB: there's effectively a copy of this function
  // in `core/svg.ts`.
  // if you change something here, you must keep it consistent there as well.
  setSvgSize(svg) {
    const { widthInPoints, heightInPoints, explicitHeight, explicitWidth } =
      this.resolveSize(svg);

    if (explicitWidth && explicitHeight) {
      svg.setAttribute("width", widthInPoints);
      svg.setAttribute("height", heightInPoints);
      svg.style.maxWidth = null; // remove mermaid's default max-width
    } else {
      if (explicitWidth) {
        svg.style.maxWidth = `${widthInPoints}px`;
      }
      if (explicitHeight) {
        svg.style.maxHeight = `${heightInPoints}px`;
      }
    }
  },

  // NB: there's effectively a copy of this function
  // in `core/svg.ts`.
  // if you change something here, you must keep it consistent there as well.
  makeResponsive(svg) {
    const width = svg.getAttribute("width");
    if (width === null) {
      throw new Error("Couldn't find SVG width");
    }
    const numWidth = Number(width.slice(0, -2));

    if (numWidth > 650) {
      changed = true;
      svg.setAttribute("width", "100%");
      svg.removeAttribute("height");
    }
  },

  // NB: there's effectively a copy of this function
  // in `core/svg.ts`.
  // if you change something here, you must keep it consistent there as well.
  fixupAlignment(svg, align) {
    let style = svg.getAttribute("style") || "";

    switch (align) {
      case "left":
        style = `${style}; display: block; margin: auto auto auto 0`;
        break;
      case "right":
        style = `${style}; display: block; margin: auto 0 auto auto`;
        break;
      case "center":
        style = `${style}; display: block; margin: auto auto auto auto`;
        break;
    }
    svg.setAttribute("style", style);
  },

  resolveOptions(svgEl) {
    return svgEl.parentElement.parentElement.parentElement.parentElement
      .dataset;
  },

  // NB: there's effectively a copy of this function
  // in our mermaid runtime in `core/svg.ts`.
  // if you change something here, you must keep it consistent there as well.
  resolveSize(svgEl) {
    const inInches = (size) => {
      if (size.endsWith("in")) {
        return Number(size.slice(0, -2));
      }
      if (size.endsWith("pt") || size.endsWith("px")) {
        // assume 96 dpi for now
        return Number(size.slice(0, -2)) / 96;
      }
      return Number(size);
    };

    // these are figWidth and figHeight on purpose,
    // because data attributes are translated to camelCase by the DOM API
    const kFigWidth = "figWidth",
      kFigHeight = "figHeight";
    const options = this.resolveOptions(svgEl);
    let width = svgEl.getAttribute("width");
    let height = svgEl.getAttribute("height");
    const getViewBox = () => {
      const vb = svgEl.attributes.getNamedItem("viewBox").value; // do it the roundabout way so that viewBox isn't dropped by deno_dom and text/html
      if (!vb) return undefined;
      const lst = vb.trim().split(" ").map(Number);
      if (lst.length !== 4) return undefined;
      if (lst.some(isNaN)) return undefined;
      return lst;
    };
    if (!width || !height) {
      // attempt to resolve figure dimensions via viewBox
      const viewBox = getViewBox();
      if (viewBox !== undefined) {
        const [_mx, _my, vbWidth, vbHeight] = viewBox;
        width = `${vbWidth}px`;
        height = `${vbHeight}px`;
      } else {
        throw new Error(
          "Mermaid generated an SVG without a viewbox attribute. Without knowing the diagram dimensions, quarto cannot convert it to a PNG"
        );
      }
    }

    let svgWidthInInches, svgHeightInInches;

    if (
      (width.slice(0, -2) === "pt" && height.slice(0, -2) === "pt") ||
      (width.slice(0, -2) === "px" && height.slice(0, -2) === "px") ||
      (!isNaN(Number(width)) && !isNaN(Number(height)))
    ) {
      // we assume 96 dpi which is generally what seems to be used.
      svgWidthInInches = Number(width.slice(0, -2)) / 96;
      svgHeightInInches = Number(height.slice(0, -2)) / 96;
    }
    const viewBox = getViewBox();
    if (viewBox !== undefined) {
      // assume width and height come from viewbox.
      const [_mx, _my, vbWidth, vbHeight] = viewBox;
      svgWidthInInches = vbWidth / 96;
      svgHeightInInches = vbHeight / 96;
    } else {
      throw new Error(
        "Internal Error: Couldn't resolve width and height of SVG"
      );
    }
    const svgWidthOverHeight = svgWidthInInches / svgHeightInInches;
    let widthInInches, heightInInches;

    if (options[kFigWidth] && options[kFigHeight]) {
      // both were prescribed, so just go with them
      widthInInches = inInches(String(options[kFigWidth]));
      heightInInches = inInches(String(options[kFigHeight]));
    } else if (options[kFigWidth]) {
      // we were only given width, use that and adjust height based on aspect ratio;
      widthInInches = inInches(String(options[kFigWidth]));
      heightInInches = widthInInches / svgWidthOverHeight;
    } else if (options[kFigHeight]) {
      // we were only given height, use that and adjust width based on aspect ratio;
      heightInInches = inInches(String(options[kFigHeight]));
      widthInInches = heightInInches * svgWidthOverHeight;
    } else {
      // we were not given either, use svg's prescribed height
      heightInInches = svgHeightInInches;
      widthInInches = svgWidthInInches;
    }

    return {
      widthInInches,
      heightInInches,
      widthInPoints: Math.round(widthInInches * 96),
      heightInPoints: Math.round(heightInInches * 96),
      explicitWidth: options?.[kFigWidth] !== undefined,
      explicitHeight: options?.[kFigHeight] !== undefined,
    };
  },

  postProcess(svg) {
    const options = this.resolveOptions(svg);
    if (
      options.responsive &&
      options["figWidth"] === undefined &&
      options["figHeight"] === undefined
    ) {
      this.makeResponsive(svg);
    } else {
      this.setSvgSize(svg);
    }
    if (options["reveal"]) {
      this.fixupAlignment(svg, options["figAlign"] || "center");
    }

    // forward align attributes to the correct parent dif
    // so that the svg figure is aligned correctly
    const div = svg.parentElement.parentElement.parentElement;
    const align = div.parentElement.parentElement.dataset.layoutAlign;
    if (align) {
      div.classList.remove("quarto-figure-left");
      div.classList.remove("quarto-figure-center");
      div.classList.remove("quarto-figure-right");
      div.classList.add(`quarto-figure-${align}`);
    }
  },
};

// deno-lint-ignore no-window-prefix
window.addEventListener(
  "load",
  async function () {
    let i = 0;
    // we need pre because of whitespace preservation
    for (const el of Array.from(document.querySelectorAll("pre.mermaid-js"))) {
      // &nbsp; doesn't appear to be treated as whitespace by mermaid
      // so we replace it with a space.
      const text = el.textContent.replaceAll("&nbsp;", " ");
      const { svg: output } = await mermaid.mermaidAPI.render(
        `mermaid-${++i}`,
        text,
        el
      );
      el.innerHTML = output;
      if (el.dataset.label) {
        // patch mermaid's emitted style
        const svg = el.firstChild;
        const style = svg.querySelector("style");
        style.innerHTML = style.innerHTML.replaceAll(
          `#${svg.id}`,
          `#${el.dataset.label}-mermaid`
        );
        svg.id = el.dataset.label + "-mermaid";
        delete el.dataset.label;
      }

      const svg = el.querySelector("svg");
      const parent = el.parentElement;
      parent.removeChild(el);
      parent.appendChild(svg);
      svg.classList.add("mermaid-js");
    }
    for (const svgEl of Array.from(
      document.querySelectorAll("svg.mermaid-js")
    )) {
      _quartoMermaid.postProcess(svgEl);
    }
  },
  false
);
