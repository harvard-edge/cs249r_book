// Theme bootstrap — runs synchronously before paint to avoid FOUC.
//
// This file MUST be loaded via a render-blocking <script src="..."> tag in
// the document <head> (no async, no defer). It reads the user's persisted
// theme preference and applies it to <html data-theme="..."> before the
// body parses, so CSS variables resolve to the right values on first paint.
//
// Lives in /public so it ships as a same-origin static asset, which lets us
// drop 'unsafe-inline' from the document CSP without losing the theme
// bootstrap. See src/app/layout.tsx for the matching <script> tag.
(function () {
  try {
    var t = localStorage.getItem("staffml_theme");
    if (t !== "light" && t !== "dark") t = "dark";
    document.documentElement.dataset.theme = t;
  } catch (_) {
    // localStorage may be unavailable (privacy mode, sandboxed iframe).
    // Fall back to the dark default that matches the SSR markup.
    document.documentElement.dataset.theme = "dark";
  }
})();
