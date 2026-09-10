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
//
// Cross-site interop:
//   The Quarto-built ecosystem subsites (book, labs, kits, slides, ...)
//   persist the user's choice under the localStorage key
//   `quarto-color-scheme`. We read that key as a secondary source when
//   StaffML has no preference of its own, so a user who toggled dark mode
//   on the book lands here in dark mode on first visit.
//
//   The matching write-back lives in the StaffML theme toggle (see
//   src/components/Providers.tsx / theme hook) — when the user toggles
//   here we mirror the choice into `quarto-color-scheme`, so navigating
//   onward to the book inherits the StaffML choice.
(function () {
  function pick() {
    try {
      var t = localStorage.getItem("staffml_theme");
      if (t === "light" || t === "dark") return t;
      // Fall back to the ecosystem-shared key set by Quarto sites.
      var q = localStorage.getItem("quarto-color-scheme");
      if (q === "light" || q === "dark") return q;
    } catch (_) {
      /* localStorage unavailable (privacy mode, sandboxed iframe). */
    }
    // Ecosystem-wide default is light (book, labs, kits, slides, etc.).
    // StaffML matches so users don't get visual whiplash crossing between
    // the book and the interview prep. Users who want dark toggle once,
    // the choice persists in localStorage AND mirrors to the Quarto sites
    // via quarto-color-scheme.
    return "light";
  }
  document.documentElement.dataset.theme = pick();
})();
