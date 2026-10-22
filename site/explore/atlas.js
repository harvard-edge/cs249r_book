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
      $("chapter-context").textContent = ch
        ? `${ch.title}: ${matches.length} concepts taught here (teal); ${revisited.length} revisited or previewed here (blue outline).`
        : "Click a concept to follow its labeled connections across chapter boundaries.";
      $("focus-chapter").disabled = !matches.length && !revisited.length;
      saveSelection();
    }
    $("chapter-selector").onchange = () => {
      chapter = $("chapter-selector").value;
      highlight();
    };
    $("focus-chapter").onclick = () =>
      focusNodes(
        data.nodes.filter((n) => {
          const ch = catalog.volumes.flatMap((v) => v.chapters).find((c) => c.id === chapter);
          return n.chapter === chapter || (ch?.references || []).some((r) => r.concept === n.teaching_goal);
        }).map((n) => n.id),
      );
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
      },
    };
  }
  return { init };
})();
