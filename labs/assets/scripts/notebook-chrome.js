/* Bridge the shared site's color scheme to native notebook classes.
 * Preference ownership remains in shared/scripts/theme-persist.js. */
(() => {
  const chrome = document.getElementById('lab-site-chrome');
  const html = document.documentElement;
  const body = document.body;
  const resize = () => html.style.setProperty('--lab-course-height', chrome.getBoundingClientRect().height + 'px');
  resize();
  new ResizeObserver(resize).observe(chrome);

  const setClass = (node, name, enabled) => {
    if (node.classList.contains(name) !== enabled) node.classList.toggle(name, enabled);
  };
  const syncTheme = () => {
    const dark = html.dataset.bsTheme === 'dark';
    const scheme = dark ? 'dark' : 'light';
    for (const node of [html, body]) {
      setClass(node, 'dark', dark);
      setClass(node, 'light', !dark);
    }
    setClass(body, 'dark-theme', dark);
    setClass(body, 'light-theme', !dark);
    if (body.dataset.theme !== scheme) body.dataset.theme = scheme;
    document.querySelectorAll('[data-lab-chrome]').forEach(link => {
      const disabled = (link.dataset.labChrome === 'dark') !== dark;
      if (link.disabled !== disabled) link.disabled = disabled;
    });
    const header = document.getElementById('quarto-header');
    setClass(header, 'quarto-dark', dark);
    setClass(header, 'quarto-light', !dark);
    const toggle = header.querySelector('.quarto-color-scheme-toggle');
    if (toggle) {
      setClass(toggle, 'alternate', dark);
      toggle.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
    }
  };
  syncTheme();
  new MutationObserver(syncTheme).observe(html, {attributes:true, attributeFilter:['data-bs-theme']});
  // Native startup can install its default body classes after the shared header.
  // Reconcile those classes without introducing another stored preference.
  new MutationObserver(syncTheme).observe(body, {attributes:true, attributeFilter:['class','data-theme']});
  window.quartoToggleColorScheme = () => {
    const dark = html.dataset.bsTheme !== 'dark';
    try { localStorage.setItem('quarto-color-scheme', dark ? 'alternate' : 'default'); } catch (_) {}
    html.dataset.bsTheme = dark ? 'dark' : 'light';
    html.dataset.quartoColorScheme = dark ? 'dark' : 'light';
    html.style.colorScheme = dark ? 'dark' : 'light';
    syncTheme();
  };
})();
