/*
 * scrollable-regions.js
 *
 * Makes scrollable code blocks and cell output keyboard-focusable while they
 * actually overflow, so keyboard users can Tab to them and scroll with the
 * arrow keys (axe scrollable-region-focusable, WCAG 2.1.1). Chrome 132+ and
 * Firefox already focus such scrollers natively; this adds Safari coverage
 * and an accessible name, and keeps the attributes in sync so a region that
 * stops overflowing stops being a tab stop.
 *
 * Copyright (C) 2026 Posit Software, PBC
 */

// Marker for attributes we added, so removal never touches author markup.
const kMarker = "data-quarto-scrollable";

// Scroll containers in rendered HTML: Pandoc emits
// `div.sourceCode { overflow: auto }` for highlighted code, the Bootstrap
// reboot gives every `pre` `overflow: auto`, and _quarto-rules.scss gives
// `.cell-output-display:not(.no-overflow-x)` `overflow-x: auto`.
const kCandidateSelector =
  "div.sourceCode, pre, .cell-output-display:not(.no-overflow-x)";

// Mirrors axe's pass condition (and Chrome's native heuristic): a region with
// keyboard-focusable content is already reachable, and adding tabindex would
// create a double tab stop. Two kinds of element look focusable but are not:
// tabindex="-1" removes an element from the tab order, so Pandoc's per-line
// anchors (`a[href][tabindex="-1"]`) don't count, and a disabled control is
// unreachable whatever its tabindex. `:disabled` matches only form controls
// (including every control inside a disabled fieldset), so it is inert on the
// generic `[tabindex]` arm and harmless there.
const kFocusableSelector =
  'a[href]:not([tabindex="-1"]), ' +
  'button:not([tabindex="-1"]):not(:disabled), ' +
  'input:not([tabindex="-1"]):not(:disabled), ' +
  'select:not([tabindex="-1"]):not(:disabled), ' +
  'textarea:not([tabindex="-1"]):not(:disabled), ' +
  '[tabindex]:not([tabindex="-1"]):not(:disabled)';

// Pandoc emits one fragment link per numbered line, as the first child of the
// line's span. They are real, focusable links and stay that way on purpose
// (#14655), but every one sits at the start of its line, so they can never
// scroll the region horizontally. A numbered block therefore satisfies axe
// while its clipped content stays unreachable — so they don't count as
// focusable content here, and a numbered block gets a region tab stop of its
// own in addition to its per-line links.
const kLineNumberAnchorSelector = "code > span > a:first-child";

// True when the region holds focusable content that already makes it
// reachable and scrollable — in which case adding tabindex would only create
// a redundant tab stop.
export function hasFocusableContent(el) {
  for (const candidate of el.querySelectorAll(kFocusableSelector)) {
    if (!candidate.matches(kLineNumberAnchorSelector)) {
      return true;
    }
  }
  return false;
}

const kDefaultLabels = {
  code: "Scrollable code",
  output: "Scrollable output",
};

// Merge configured labels over the English defaults, ignoring missing or
// non-string values (e.g. a language file without the new keys).
export function resolveLabels(labels) {
  const resolved = { ...kDefaultLabels };
  for (const key of Object.keys(kDefaultLabels)) {
    if (labels && typeof labels[key] === "string" && labels[key] !== "") {
      resolved[key] = labels[key];
    }
  }
  return resolved;
}

// A region scrolls when it is a scroll container (computed overflow auto or
// scroll — axe's own condition) whose content overflows its box on that axis;
// the 1px tolerance absorbs subpixel rounding. Without the overflow check, a
// child that merely bleeds outside a scrolling parent (pre.sourceCode inside
// div.sourceCode has overflow visible but the same scrollWidth) would become
// a second, useless tab stop. Pure over hand-built geometry and style.
export function isScrollable(el, style) {
  const scrollsX = style.overflowX === "auto" || style.overflowX === "scroll";
  const scrollsY = style.overflowY === "auto" || style.overflowY === "scroll";
  return (
    (scrollsX && el.scrollWidth - el.clientWidth > 1) ||
    (scrollsY && el.scrollHeight - el.clientHeight > 1)
  );
}

// A region needs a usable size before a user can scroll it. Closed <details>
// and inactive tab panes report 0. Visually-hidden alternatives are clipped to
// 1px (`.visually-hidden` sets width 1px and overflow hidden; CSS then computes
// the other axis to auto), so their full content height reads as overflow —
// marking one would add an invisible tab stop.
export function hasUsableSize(el) {
  return el.clientWidth > 1 && el.clientHeight > 1;
}

// Cleanup is skipped while a region has focus, so attributes never come out
// from under the user mid-interaction. Nothing else would rerun it afterwards
// though, so a region that stopped overflowing while focused would stay a tab
// stop until the next unrelated resize. A one-shot `focusout` listener closes
// that gap, and the set keeps repeated syncs from stacking listeners on the
// same element.
const kPendingFocusOut = new WeakSet();

function resyncWhenFocusLeaves(el, labels) {
  if (kPendingFocusOut.has(el)) {
    return;
  }
  kPendingFocusOut.add(el);
  el.addEventListener(
    "focusout",
    () => {
      kPendingFocusOut.delete(el);
      syncScrollableRegions(labels);
    },
    { once: true }
  );
}

function labelFor(el, labels) {
  if (el.matches(".cell-output-display") || el.closest(".cell-output")) {
    return labels.output;
  }
  return labels.code;
}

export function syncScrollableRegions(labels) {
  labels = resolveLabels(labels || window.quartoScrollableRegionsLabels);
  for (const el of document.querySelectorAll(kCandidateSelector)) {
    if (!hasUsableSize(el)) {
      continue;
    }
    if (!el.hasAttribute(kMarker)) {
      // Respect author markup and regions with their own tab stops. `role`
      // counts: cleanup removes the three attributes outright, so marking an
      // element that already had one would let a later re-sync strip it.
      if (
        el.hasAttribute("tabindex") ||
        el.hasAttribute("role") ||
        el.hasAttribute("aria-label") ||
        hasFocusableContent(el)
      ) {
        continue;
      }
      if (isScrollable(el, getComputedStyle(el))) {
        el.setAttribute("tabindex", "0");
        el.setAttribute("role", "group");
        el.setAttribute("aria-label", labelFor(el, labels));
        el.setAttribute(kMarker, "");
      }
    } else if (!isScrollable(el, getComputedStyle(el))) {
      if (el === document.activeElement) {
        resyncWhenFocusLeaves(el, labels);
        continue;
      }
      el.removeAttribute("tabindex");
      el.removeAttribute("role");
      el.removeAttribute("aria-label");
      el.removeAttribute(kMarker);
    }
  }
}
