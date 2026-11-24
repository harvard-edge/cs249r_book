-- Shared closing block for the volume landing pages.
-- Usage: {{< volume-closing >}}
--
-- Renders the GitHub star card (goal, star count, Star button, Open Collective)
-- and the Contributing section, so all four volume homepages stay identical.
-- Edit the text here, not in the pages.
--
-- The star count is hidden until GitHub's API answers. The count is cached in
-- sessionStorage for an hour, so moving between volumes does not refetch, and a
-- failed or rate-limited fetch leaves only the Star button instead of a stuck
-- "Loading..." label.

local REPO_URL = "https://github.com/harvard-edge/cs249r_book"
local API_URL = "https://api.github.com/repos/harvard-edge/cs249r_book"

local HTML = [[
<div class="support-mission">
  <p><strong>Goal:</strong> Help 100,000 students learn machine learning systems. Sponsors like the <a href="https://edgeaifoundation.org/" target="_blank" rel="noopener noreferrer">EDGE AI Foundation</a> match every star with funding that supports learning.</p>

  <div class="support-actions">
    <span class="star-count" data-github-stars hidden></span>
    <a href="]] .. REPO_URL .. [[" target="_blank" rel="noopener" class="github-star-btn">Star on GitHub</a>
  </div>

  <p class="support-note">
    <a href="https://opencollective.com/mlsysbook" target="_blank" rel="noopener">Support us on Open Collective →</a>
  </p>
</div>

<script>
(function () {
  var els = document.querySelectorAll("[data-github-stars]");
  if (!els.length) return;
  var KEY = "mlsysbook-github-stars";
  function show(n) {
    var text = n >= 1000 ? (n / 1000).toFixed(1).replace(/\.0$/, "") + "k" : String(n);
    els.forEach(function (el) {
      el.textContent = text;
      el.title = n.toLocaleString() + " GitHub stars";
      el.hidden = false;
    });
  }
  try {
    var cached = JSON.parse(sessionStorage.getItem(KEY) || "null");
    if (cached && typeof cached.n === "number" && Date.now() - cached.t < 3600000) {
      show(cached.n);
      return;
    }
  } catch (e) {}
  fetch("]] .. API_URL .. [[")
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (d) {
      if (!d || typeof d.stargazers_count !== "number") return;
      show(d.stargazers_count);
      try { sessionStorage.setItem(KEY, JSON.stringify({ n: d.stargazers_count, t: Date.now() })); } catch (e) {}
    })
    .catch(function () {});
})();
</script>

<h2>Contributing</h2>

<p>This is a collaborative project. The project <a href="]] .. REPO_URL .. [[/blob/main/CONTRIBUTING.md">contribution guidelines</a> explain how to participate. Feedback, corrections, and new ideas are welcome through GitHub <a href="]] .. REPO_URL .. [[/issues">issues</a>.</p>
]]

return {
  ["volume-closing"] = function(args, kwargs, meta)
    if not quarto.doc.is_format("html") then
      return pandoc.Null()
    end
    return pandoc.RawBlock("html", HTML)
  end
}
