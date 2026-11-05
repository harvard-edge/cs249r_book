// Keep the existing study workspace out of the initial focused-reader bundle.
// Resolve beside the reader itself so every volume and deployment prefix works.
export function createStudyTools({scriptUrl, document: doc = document, window: win = window} = {}) {
  const url = new URL('study.js', scriptUrl || doc.baseURI).href;
  let loading = null;

  function load() {
    if (win.SocratiqStudy) return Promise.resolve(win.SocratiqStudy);
    if (loading) return loading;
    win.__SOCRATIQ_EMBEDDED__ = true;
    loading = new Promise((resolve, reject) => {
      const script = doc.createElement('script');
      // The existing Vite workspace entry uses import.meta for bundled assets.
      script.type = 'module';
      script.src = url;
      script.async = true;
      script.dataset.socratiqStudy = '';
      const timer = win.setTimeout(() => fail(), 30000);
      function fail() {
        win.clearTimeout(timer);
        script.remove();
        loading = null;
        reject(new Error('Study tools could not load. Check your connection and try again.'));
      }
      script.onerror = fail;
      script.onload = () => {
        win.clearTimeout(timer);
        if (!win.SocratiqStudy?.open) { fail(); return; }
        resolve(win.SocratiqStudy);
      };
      doc.head.appendChild(script);
    });
    return loading;
  }

  return {
    async open() {
      const tools = await load();
      // open() owns bootstrap readiness and can retry a failed initialization.
      await tools.open();
    },
  };
}
