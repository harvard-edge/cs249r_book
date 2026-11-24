/* A chapter highlights a region of the volume graph; it never replaces it. */
window.MLSysVolume = (() => {
  const $ = (id) => document.getElementById(id);
  const esc = (v) =>
    String(v).replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  function init(catalog, openMap, focusNodes, saveSelection) {
    let autoFocused = false;
    let data = null;
    const state = new URLSearchParams(location.hash.slice(1));
    let chapter =
      new URLSearchParams(location.search).get("chapter") ||
      state.get("chapter") ||
      catalog.chapter ||
      "";
    const metadata = new Map(catalog.volumes.map((v) => [v.id, v]));
    const overview = catalog.maps.filter((m) => m.scope === "volume");
    $("volume-buttons").innerHTML = catalog.volumes
      .map(
        (v) =>
          `<button type="button" data-volume="${esc(v.id)}" aria-pressed="false"><span>VOLUME ${esc(v.label || v.id.slice(3))}</span><strong>${esc(v.scope || v.id)}</strong><small>${esc(v.title || "")}</small></button>`,
      )
      .join("");
    $("volume-buttons")
      .querySelectorAll("button")
      .forEach((b) => {
        const map = overview.find((m) =>
          m.chapter.startsWith(b.dataset.volume + "/"),
        );
        b.disabled = !map;
        b.onclick = () => {
          chapter = "";
          openMap(map.id);
        };
      });
    function highlight() {
      if (!data) return;
      const ch = catalog.volumes.flatMap((v) => v.chapters).find((c) => c.id === chapter);
      const referenced = new Set((ch?.references || []).map((r) => r.concept));
      const matches = data.nodes.filter((n) => n.chapter === chapter);
      const revisited = data.nodes.filter((n) => n.chapter !== chapter && referenced.has(n.teaching_goal));
      for (const n of data.nodes) {
        const el = document.getElementById("concept-" + n.id);
        el?.classList.toggle("chapter-focus", n.chapter === chapter);
        el?.classList.toggle("chapter-reference", n.chapter !== chapter && referenced.has(n.teaching_goal));
      }
      // A chapter's concepts are spread across the volume map, so at a readable
      // size they rarely fit in one view. Naming them, each a link that pans to
      // it, keeps every one a click away without rearranging the map.
      const context = $("chapter-context");
      if (ch) {
        const names = matches
          .map((n) => `<button type="button" class="locate-concept" data-concept="${esc(n.id)}">${esc(n.label)}</button>`)
          .join(", ");
        const taughtText = matches.length
          ? `${esc(ch.title)} teaches ${matches.length === 1 ? "one concept" : `${matches.length} concepts`} here (teal): ${names}.`
          : `${esc(ch.title)} introduces no concepts of its own in this map.`;
        const revisitedText = revisited.length
          ? ` It revisits or previews ${revisited.length} more (blue outline).`
          : "";
        context.innerHTML = taughtText + revisitedText;
        context.querySelectorAll(".locate-concept").forEach((b) => {
          b.onclick = () => {
            focusNodes([b.dataset.concept], null, true);
            const el = document.getElementById("concept-" + b.dataset.concept);
            el?.classList.remove("located");
            void el?.getBoundingClientRect();
            el?.classList.add("located");
          };
        });
      } else {
        context.textContent = "Click a concept to follow its labeled connections across chapter boundaries.";
      }
      $("focus-chapter").disabled = !matches.length && !revisited.length;
      saveSelection();
    }
    $("chapter-selector").onchange = () => {
      chapter = $("chapter-selector").value;
      highlight();
    };
    // The chapter's concepts: those it teaches, and those it revisits or previews.
    function chapterNodes() {
      const ch = catalog.volumes.flatMap((v) => v.chapters).find((c) => c.id === chapter);
      const referenced = new Set((ch?.references || []).map((r) => r.concept));
      const taught = data.nodes.filter((n) => n.chapter === chapter).map((n) => n.id);
      const revisited = data.nodes
        .filter((n) => n.chapter !== chapter && referenced.has(n.teaching_goal))
        .map((n) => n.id);
      return { taught, all: [...taught, ...revisited] };
    }
    // Fit everything the chapter touches; when that would shrink labels below
    // readable size, open on the part of the map with the most taught concepts.
    $("focus-chapter").onclick = () => {
      const { taught, all } = chapterNodes();
      focusNodes(all.length ? all : taught, taught);
    };
    return {
      chapter: () => chapter,
      mapLoaded(next) {
        data = next;
        const volume = data.chapter.split("/")[0],
          v = metadata.get(volume);
        if (chapter && !chapter.startsWith(volume + "/")) chapter = "";
        $("chapter-selector").innerHTML =
          '<option value="">Whole volume</option>' +
          v.chapters
            .filter((ch) => ch.concepts.length)
            .map(
              (ch) => `<option value="${esc(ch.id)}">${esc(ch.title)}</option>`,
            )
            .join("");
        $("chapter-selector").value = chapter;
        $("volume-buttons")
          .querySelectorAll("button")
          .forEach((b) =>
            b.setAttribute("aria-pressed", String(b.dataset.volume === volume)),
          );
        highlight();
        // Opened for a chapter (inside it, or from its "Open the full map"
        // link), start on that chapter's concepts rather than the whole
        // volume, which is unreadable at a chapter's size. focusNodes keeps
        // labels readable and favors the concepts the chapter teaches.
        if (!autoFocused && chapter) {
          autoFocused = true;
          setTimeout(() => $("focus-chapter").click(), 60);
        }
      },
    };
  }
  return { init };
})();
