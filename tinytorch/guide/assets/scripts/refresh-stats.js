<script>
// Keep the GitHub star count current after deploy.
//
// inject_stats.py substitutes {{stats.stars}} at render time, so the number is
// correct at the moment the site is built and then frozen. The main site solves
// this in site/index.qmd by re-fetching on every page load; this site emitted
// the same data-stat="stars" hook but shipped nothing to hydrate it, so a
// CDN-cached page showed whatever the cache held at the last publish. The
// cached value was 152 stars behind within three days.
//
// The star count is public and needs no credential, so it is fetched directly
// from the API rather than waiting on the six-hourly stats.json refresh, which
// this site does not consume. On any failure the build-time value simply
// stays, which is why the elements are never emptied first.
(function () {
  var nodes = document.querySelectorAll('[data-stat="stars"]');
  if (!nodes.length) return;

  fetch('https://api.github.com/repos/harvard-edge/cs249r_book')
    .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
    .then(function (d) {
      if (!d || typeof d.stargazers_count !== 'number') return;
      var formatted = d.stargazers_count.toLocaleString('en-US');
      nodes.forEach(function (n) { n.textContent = formatted; });
    })
    .catch(function (err) {
      // Rate limit, offline, blocked: the rendered value is still a real number.
      console.warn('[tinytorch] star count refresh failed, showing build-time value:', err);
    });
})();
</script>
