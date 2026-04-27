/* =============================================================================
 * theme-persist.js — cross-site dark/light mode persistence + FOUC guard
 * =============================================================================
 *
 * Why this exists
 * ---------------
 * Every subsite under mlsysbook.ai is built by a different engine (Quarto for
 * the book/labs/kits/slides/site/instructors/tinytorch, Next.js for StaffML,
 * etc). Quarto's built-in toggle stores the user's choice in localStorage
 * under the key `quarto-color-scheme`. Because every subsite shares the
 * mlsysbook.ai origin in production, that localStorage entry is *technically*
 * shared — but Quarto's own bridge script runs late, after the page has
 * already painted in the wrong theme. The result is a visible flash when a
 * dark-mode reader navigates from one subsite to another.
 *
 * This script:
 *   1. Runs as the FIRST <script> in <head> (so it executes before render).
 *   2. Reads the user's stored color-scheme preference and applies the
 *      corresponding `data-bs-theme` / `data-quarto-color-scheme` attributes
 *      on <html> *before* the browser paints, eliminating the flash.
 *   3. Falls back to OS preference (`prefers-color-scheme: dark`) when the
 *      user has not yet expressed a preference.
 *   4. Listens for `storage` events so a toggle in tab A is reflected
 *      immediately in tab B without a refresh.
 *
 * It deliberately does *not* render any UI: Quarto's existing toggle button
 * stays the source of truth for user-initiated changes; this script only
 * preempts the flash and synchronizes other open tabs.
 *
 * Wiring (Quarto sites): add to include-in-header BEFORE any other script:
 *   <script src="/assets/scripts/theme-persist.js"></script>
 *
 * Wiring (non-Quarto sites, e.g. StaffML/Next.js): drop the same file at the
 * same URL and reference it from the document head. The script is framework-
 * agnostic; it only touches `<html>` attributes and a single localStorage key.
 *
 * Canonical source: shared/scripts/theme-persist.js
 * Mirrors:          synced via shared/scripts/sync-mirrors.sh (do not hand-edit)
 * ============================================================================= */
(function () {
  'use strict';

  // The single key both Quarto and this shim agree on. Do NOT rename without
  // also updating Quarto's expected key — Quarto reads/writes the same value
  // and we explicitly want to interoperate, not shadow it.
  var STORAGE_KEY = 'quarto-color-scheme';

  // Quarto's built-in toggle stores 'alternate' (= dark, the layered sheet)
  // or 'default' (= light, the base sheet) under STORAGE_KEY. Sibling
  // (non-Quarto) subsites store 'dark' / 'light'. Both conventions hit the
  // same key, so we accept either and normalize. Quarto's own startup still
  // checks `=== 'alternate'`, so we MUST NOT rewrite Quarto's writes — only
  // mirror them onto <html>'s data-bs-theme attribute.
  function normalize(value) {
    if (value === 'dark' || value === 'alternate') return 'dark';
    if (value === 'light' || value === 'default') return 'light';
    return null;
  }

  function preferredScheme() {
    try {
      var n = normalize(window.localStorage.getItem(STORAGE_KEY));
      if (n) return n;
    } catch (e) {
      // localStorage may be blocked (private mode, sandboxed iframe).
      // Silent fallback to OS preference is the right behavior.
    }
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    return 'light';
  }

  function apply(scheme) {
    var html = document.documentElement;
    if (!html) return;
    // Bootstrap 5.3+: data-bs-theme drives semantic colors.
    html.setAttribute('data-bs-theme', scheme);
    // Quarto-internal hook used by some subsite SCSS (kept in sync to avoid
    // half-themed elements during the gap before Quarto's own script runs).
    html.setAttribute('data-quarto-color-scheme', scheme);
    // Hint for any CSS using the @media `prefers-color-scheme` shorthand.
    html.style.colorScheme = scheme;
  }

  // 1. Apply the initial scheme synchronously (before paint).
  apply(preferredScheme());

  // 2. Same-tab sync: Quarto's toggle writes to localStorage but does NOT
  //    update <html> attributes. After Quarto's toggle handler runs, mirror
  //    the new storage value onto <html>. Without this, clicking the toggle
  //    leaves data-bs-theme stale until the next reload, and any CSS keyed
  //    off [data-bs-theme="dark"] renders against the wrong stylesheet.
  function syncFromStorage() {
    try {
      var n = normalize(window.localStorage.getItem(STORAGE_KEY));
      if (n) apply(n);
    } catch (e) { /* see preferredScheme */ }
  }
  function wrapToggle() {
    var orig = window.quartoToggleColorScheme;
    if (typeof orig !== 'function' || orig.__mlsbWrapped) return;
    var wrapped = function () {
      var r = orig.apply(this, arguments);
      syncFromStorage();
      return r;
    };
    wrapped.__mlsbWrapped = true;
    window.quartoToggleColorScheme = wrapped;
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      wrapToggle();
      // Quarto's startup may pick a different scheme than us if storage held
      // an 'alternate'/'default' Quarto value we previously couldn't parse;
      // reconcile once Quarto has finished its own startup.
      syncFromStorage();
    });
  } else {
    wrapToggle();
    syncFromStorage();
  }

  // 3. Cross-tab sync: react when *another* tab updates the choice.
  window.addEventListener('storage', function (ev) {
    if (ev.key !== STORAGE_KEY) return;
    var n = normalize(ev.newValue);
    if (n) apply(n);
  });

  // 4. OS-preference change: only apply if the user has NOT explicitly
  //    chosen via the toggle (i.e. nothing in localStorage).
  if (window.matchMedia) {
    try {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function (ev) {
        try {
          if (window.localStorage.getItem(STORAGE_KEY)) return;
        } catch (e) {
          /* If we can't read storage, treating OS pref as truth is fine. */
        }
        apply(ev.matches ? 'dark' : 'light');
      });
    } catch (e) {
      // Older Safari does not support addEventListener on MediaQueryList.
      // The initial apply() above is sufficient for those browsers.
    }
  }
})();
