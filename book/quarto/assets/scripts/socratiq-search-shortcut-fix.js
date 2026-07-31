// Frees `Ctrl/Cmd + /` for SocratiQ.
//
// Quarto's built-in search binds the `/` key on `keyup` and checks ONLY
// `event.key === "/"`, ignoring modifier keys (see quarto-search.js). So
// `Ctrl/Cmd + /` — SocratiQ's documented shortcut — also matches, and Quarto
// opens its search box on `keyup`. SocratiQ's own handler fires on `keydown`
// and calls preventDefault(), but that does not stop Quarto's separate `keyup`
// listener.
//
// This shim intercepts `/` in the capture phase and, when a modifier is held,
// calls stopImmediatePropagation() so Quarto's keyup handler never runs. Plain
// `/` is untouched, so Quarto search still works as before. SocratiQ's keydown
// handler then opens the chat unopposed (only when SocratiQ is enabled).
(function () {
  function isModifierSlash(e) {
    return e.key === "/" && (e.ctrlKey || e.metaKey);
  }
  // Capture phase = runs before Quarto's bubble-phase keyup listener.
  document.addEventListener(
    "keyup",
    function (e) {
      if (isModifierSlash(e)) {
        e.stopImmediatePropagation();
      }
    },
    true
  );
})();
