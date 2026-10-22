/* Deterministic, diagram-first mechanism scenes. Snapshot data owns meaning;
 * browser animation only illustrates one transition between valid snapshots. */
(() => {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  let serial = 0;
  const node = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const svg = (tag, attrs = {}) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    return n;
  };
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  const formatNumber = (v) => {
    const rounded = Number(v.toPrecision(6)),
      changed = rounded !== v;
    let display;
    if (rounded !== 0 && (Math.abs(rounded) >= 1e7 || Math.abs(rounded) < 1e-4))
      display = rounded.toExponential().replace("e+", "e");
    else display = String(rounded);
    return { display: (changed ? "≈" : "") + display, changed };
  };
  const valueText = (v) =>
    v === null
      ? "—"
      : typeof v === "number"
        ? formatNumber(v).display
        : String(v);
  const leaf = (v, cls = "mechanism-value") => {
    const n = node("span", cls, valueText(v));
    n.dataset.value = v === null ? "null" : String(v);
    if (typeof v === "number") {
      n.dataset.numeric = "true";
      if (formatNumber(v).changed) n.title = `Exact stored value: ${String(v)}`;
    }
    if (typeof v === "string" && /^K\d+\/V\d+$/.test(v)) {
      const parts = v.split("/");
      n.replaceChildren(
        document.createTextNode(parts[0] + "/"),
        document.createElement("wbr"),
        document.createTextNode(parts[1]),
      );
      n.dataset.paired = "true";
    }
    if (v === null) n.setAttribute("aria-label", "Not yet produced");
    return n;
  };
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons["mechanism-scene"] = async (root, context = {}) => {
    const outer = context.outer,
      key = outer?.dataset.mlsysExploreKey;
    const movementMs = Number(outer?.dataset.mlsysMovementMs),
      readingPauseMs = Number(outer?.dataset.mlsysReadingPauseMs);
    if (!Number.isInteger(movementMs) || movementMs < 1 || movementMs > 60000 ||
        !Number.isInteger(readingPauseMs) || readingPauseMs < 0 || readingPauseMs > 60000)
      throw new Error("Missing or invalid shared playback timing.");
    const envelope = window.MLSysMechanismLessons?.[key],
      enrolled = outer?.dataset.mlsysLessonSha1;
    if (
      !envelope?.canonical ||
      !enrolled ||
      envelope.sha1 !== enrolled ||
      !crypto.subtle
    )
      throw new Error("Missing reviewed mechanism data.");
    const bytes = new TextEncoder().encode(envelope.canonical);
    const digest = [
      ...new Uint8Array(await crypto.subtle.digest("SHA-1", bytes)),
    ]
      .map((x) => x.toString(16).padStart(2, "0"))
      .join("");
    if (digest !== enrolled)
      throw new Error("Mechanism data changed since review.");
    const lesson = JSON.parse(envelope.canonical);
    if (
      lesson.schema !== 1 ||
      lesson.key !== key ||
      !lesson.objects?.length ||
      !lesson.steps?.length
    )
      throw new Error("Invalid mechanism identity.");
    const abort = new AbortController(),
      on = (el, event, fn) =>
        el.addEventListener(event, fn, { signal: abort.signal });
    const component = node("section", "mlsys-mechanism");
    component.style.setProperty(
      "--mechanism-narrow-columns",
      lesson.narrowColumns || 1,
    );
    component.tabIndex = 0;
    component.setAttribute("aria-label", lesson.title);
    const title = node("p", "mechanism-question", lesson.question),
      scope = node("p", "mechanism-scope");
    const controls = node("div", "mechanism-controls");
    controls.setAttribute("aria-label", "Animation controls");
    const buttons = {};
    for (const [id, label, keyLabel] of [
      ["next", "Next", "N"],
      ["previous", "Back", "P"],
      ["play", "Play", "K"],
      ["restart", "Restart", "R"],
      ["expand", "↗", ""],
    ]) {
      const b = node("button", "", label);
      b.type = "button";
      b.dataset.mechanismControl = id;
      b.setAttribute(
        "aria-label",
        id === "next"
          ? "Next action"
          : id === "previous"
            ? "Previous action"
            : id === "expand"
              ? "Expand animation"
              : label,
      );
      if (keyLabel) b.setAttribute("aria-keyshortcuts", keyLabel);
      buttons[id] = b;
      controls.append(b);
    }
    const actionCue = node("span", "mechanism-action-cue");
    actionCue.setAttribute("aria-hidden", "true");
    controls.append(actionCue);
    const selectors = node("div", "mechanism-selectors"),
      sequenceLabel = node("label", "", "Sequence "),
      sequence = node("select");
    sequence.setAttribute("aria-label", "Sequence");
    sequenceLabel.append(sequence);
    selectors.append(sequenceLabel);
    const caseLabel = node("label", "", "Case "),
      caseMenu = node("select");
    caseMenu.setAttribute("aria-label", "Case");
    const cases = [
      {
        id: "base",
        label: "Original case",
        scope: lesson.scope,
        steps: lesson.steps,
      },
      ...(lesson.variants || []),
    ];
    for (const c of cases) {
      const o = node("option", "", c.label);
      o.value = c.id;
      caseMenu.append(o);
    }
    if (cases.length > 1) {
      caseLabel.append(caseMenu);
      selectors.append(caseLabel);
    }
    const shortcut = node("div", "mechanism-shortcuts");
    shortcut.append(
      node(
        "p",
        "",
        "Keyboard (focus this animation): N Next · P Back · K Play/Pause · R Restart. Next performs one action, then pauses.",
      ),
    );
    const action = node("div", "mechanism-action"),
      heading = node("strong"),
      explanation = node("p");
    action.append(heading, explanation);
    action.setAttribute("aria-live", "polite");
    action.setAttribute("aria-atomic", "true");
    const scene = node("div", "mechanism-scene"),
      grid = node("div", "mechanism-grid"),
      overlay = node("div", "mechanism-overlay");
    overlay.setAttribute("aria-hidden", "true");
    scene.append(grid, overlay);
    const result = node("p", "mechanism-result");
    result.setAttribute("aria-live", "polite");
    const metrics = node("dl", "mechanism-metrics");
    metrics.setAttribute("aria-label", "Secondary resource measures");
    const motionNote = node(
      "p",
      "mechanism-motion-note",
      "Movement illustrates the action; its duration does not represent execution time.",
    );
    shortcut.append(motionNote);
    component.append(
      title,
      scope,
      controls,
      shortcut,
      selectors,
      action,
      scene,
      result,
      metrics,
    );
    if (lesson.details) {
      const d = node("details", "mechanism-details");
      d.append(node("summary", "", lesson.details.summary));
      for (const p of lesson.details.paragraphs) d.append(node("p", "", p));
      component.append(d);
    }
    let prediction = null;
    if (lesson.prediction) {
      prediction = node("fieldset", "mechanism-prediction");
      prediction.append(node("legend", "", lesson.prediction.question));
      const feedback = node("p");
      feedback.setAttribute("aria-live", "polite");
      for (const answer of lesson.prediction.answers) {
        const b = node("button", "", answer.label);
        b.type = "button";
        on(b, "click", () => {
          feedback.textContent = answer.feedback;
          b.setAttribute(
            "aria-pressed",
            String(answer.id === lesson.prediction.correct),
          );
        });
        prediction.append(b);
      }
      prediction.append(feedback);
      component.append(prediction);
    }
    root.append(component);
    root.classList.add("mlsys-mechanism-panel");
    context.figure?.classList.add("mlsys-mechanism-figure");
    const dialog = node("dialog", "mechanism-dialog"),
      dialogHeader = node("div", "mechanism-dialog-header"),
      closeButton = node("button", "", "Close workspace");
    closeButton.type = "button";
    dialogHeader.append(node("strong", "", lesson.title), closeButton);
    dialog.append(dialogHeader);
    document.body.append(dialog);
    const placeholder = document.createComment("mechanism workspace position");
    const media = matchMedia("(prefers-reduced-motion: reduce)");
    let currentCase = cases[0],
      steps = currentCase.steps,
      index = 0,
      committed = 0,
      moving = false,
      paused = true,
      autoplay = false,
      destroyed = false,
      epoch = 0;
    let effects = [],
      motionFinished = null,
      holdTimer = null,
      holdStarted = 0,
      holdRemaining = 0,
      layoutFrame = 0,
      expanded = false;
    let lastNarrow = false,
      stickyFrame = 0;
    const objects = new Map(lesson.objects.map((o) => [o.id, o]));
    // Reserve actual receiving surfaces before a transfer starts. The spaces
    // are UI annotations, never additional stored allocations or committed data.
    const incomingPlans = new Map(),
      receivingPlans = new Map();
    let leftCorridor = false,
      rightCorridor = false;
    function incomingValues(before, after, t, id) {
      const source = t.from && before.objects[t.from.object];
      const values =
        source?.values || source?.slots?.map((slot) => slot?.value ?? null);
      return t.kind === "execute" && t.to?.object === id && values?.length <= 8
        ? values
        : [after.objects[id].value];
    }
    function receivePlan(object, index, value, formula = "") {
      if (index === undefined) return;
      let map = receivingPlans.get(object);
      if (!map) {
        map = new Map();
        receivingPlans.set(object, map);
      }
      const old = map.get(index) || { length: 0, formula: false };
      old.length = Math.max(
        old.length,
        String(formula || valueText(value)).length,
      );
      old.formula = old.formula || Boolean(formula);
      map.set(index, old);
    }
    function noteRoute(from, to) {
      if (!from || !to || from.object === to.object) return;
      if (
        objects.get(from.object)?.layout.narrow.row !==
        objects.get(to.object)?.layout.narrow.row
      )
        leftCorridor = true;
    }
    for (const c of cases)
      for (let i = 1; i < c.steps.length; i++) {
        const before = c.steps[i - 1].snapshot,
          after = c.steps[i].snapshot,
          t = c.steps[i].transition;
        noteRoute(t.from, t.to);
        for (const o of lesson.objects.filter((o) => o.kind === "register"))
          if (
            !same(before.objects[o.id].value, after.objects[o.id].value) ||
            t.to?.object === o.id
          ) {
            const values = incomingValues(before, after, t, o.id),
              old = incomingPlans.get(o.id) || { capacity: 1, length: 1 };
            old.capacity = Math.max(old.capacity, values.length);
            old.length = Math.max(
              old.length,
              ...values.map((v) => valueText(v).length),
            );
            incomingPlans.set(o.id, old);
            if (t.to?.object === o.id && t.from) rightCorridor = true;
          }
        for (const token of after.tokens) {
          const old = before.tokens.find((x) => x.id === token.id),
            from =
              old && !same(old.at, token.at)
                ? old.at
                : token.sourceAt ||
                  before.tokens.find((x) => x.id === token.sourceId)?.at;
          if (!from) continue;
          noteRoute(from, token.at);
          receivePlan(
            token.at.object,
            token.at.index,
            token.value,
            token.state === "inflight" && token.label.includes("→")
              ? token.label
              : "",
          );
        }
        if (t.to && objects.get(t.to.object)?.kind !== "register") {
          const a = before.objects[t.to.object],
            b = after.objects[t.to.object],
            av = a.values || a.slots?.map((x) => x?.value ?? null),
            bv = b.values || b.slots?.map((x) => x?.value ?? null);
          if (bv)
            bv.forEach((v, j) => {
              if (!same(av[j], v)) receivePlan(t.to.object, j, v);
            });
        }
      }
    component.classList.toggle("mechanism-route-left", leftCorridor);
    component.classList.toggle("mechanism-route-right", rightCorridor);
    const segmentOrder = new Map();
    for (const c of cases)
      for (const step of c.steps)
        for (const o of lesson.objects.filter((o) => o.kind === "lanes"))
          for (const segment of step.snapshot.objects[o.id].segments) {
            const id = o.id + "|" + segment.id;
            if (!segmentOrder.has(id))
              segmentOrder.set(id, segmentOrder.size + 1);
          }
    const narrow = () => component.getBoundingClientRect().width < 570;
    const all = (selector) => [...grid.querySelectorAll(selector)];
    const objectNode = (id) =>
      all("[data-object-id]").find((n) => n.dataset.objectId === id);
    function currentSnapshot() {
      return steps[committed].snapshot;
    }
    function updateDataset() {
      Object.assign(component.dataset, {
        mechanismKey: key,
        mechanismNumberFormat: "6sig",
        mechanismVariant: currentCase.id,
        mechanismStep: String(index),
        mechanismStepId: steps[index].id,
        mechanismCommittedStep: String(committed),
        mechanismMoving: String(moving),
        mechanismRunning: String(!paused && autoplay),
        mechanismAutoplay: String(autoplay),
        mechanismMotionPaused: String(paused),
      });
      buttons.next.disabled = !moving && index === steps.length - 1;
      buttons.previous.disabled = index === 0;
      buttons.restart.disabled = !moving && index === 0;
      buttons.next.textContent = moving ? "Finish" : "Next";
      buttons.next.setAttribute(
        "aria-label",
        moving ? "Finish action" : "Next action",
      );
      const live = !paused && (moving || autoplay);
      buttons.play.textContent = live
        ? "Pause"
        : index === steps.length - 1 && !moving
          ? "Replay"
          : "Play";
      buttons.play.setAttribute("aria-label", buttons.play.textContent);
      sequence.value = String(index);
      heading.textContent = `${index + 1} / ${steps.length} · ${steps[index].title}`;
      explanation.textContent = steps[index].explanation;
      actionCue.textContent = `${index + 1}/${steps.length} · ${steps[index].title}${moving ? (paused ? " · paused" : " · moving") : ""}`;
      if (prediction)
        prediction.hidden = steps[index].id !== lesson.prediction.before;
      component.classList.toggle("mechanism-is-moving", moving);
      component.classList.toggle("mechanism-is-paused", paused);
    }
    function menu() {
      sequence.replaceChildren();
      steps.forEach((s, i) => {
        const o = node("option", "", `${i + 1}. ${s.title}`);
        o.value = String(i);
        sequence.append(o);
      });
      scope.textContent = currentCase.scope;
    }
    function cell(value, i, label) {
      const c = node("div", "mechanism-cell");
      c.dataset.cellIndex = String(i);
      if (label !== undefined)
        c.append(node("small", "mechanism-cell-index", label));
      c.append(leaf(value));
      c.classList.toggle("mechanism-empty", value === null);
      return c;
    }
    function axisText(parent, text, cls) {
      parent.append(node("span", cls, text));
    }
    function chart(o, s, numberLine = false, availableWidth = 400) {
      const box = node(
          "div",
          numberLine ? "mechanism-number-line" : "mechanism-plot",
        ),
        height = numberLine ? 170 : 220,
        width = Math.max(1, availableWidth || 400);
      box.style.height = `${height}px`;
      const drawing = svg("svg", {
        viewBox: "0 0 400 220",
        "aria-hidden": "true",
        preserveAspectRatio: "none",
      });
      box.append(drawing);
      const xr = numberLine
          ? { min: o.config.min, max: o.config.max }
          : o.config.x,
        yr = numberLine ? { min: 0, max: 1 } : o.config.y;
      let left = 32,
        right = 376,
        top = 25,
        bottom = 190;
      if (!numberLine && o.config.equalAxes) {
        const unit = Math.min(
          (width * 0.86) / (xr.max - xr.min),
          (height * 0.75) / (yr.max - yr.min),
        );
        const dx = (((xr.max - xr.min) * unit) / width) * 400,
          dy = (((yr.max - yr.min) * unit) / height) * 220;
        left = 32 + (344 - dx) / 2;
        right = left + dx;
        top = 25 + (165 - dy) / 2;
        bottom = top + dy;
      }
      const X = (x) =>
          left + ((x - xr.min) / (xr.max - xr.min)) * (right - left),
        Y = (y) =>
          numberLine
            ? 112
            : bottom - ((y - yr.min) / (yr.max - yr.min)) * (bottom - top);
      box.mechanismGeometry = { X, Y };
      drawing.append(
        svg("line", {
          x1: left,
          y1: numberLine ? 112 : bottom,
          x2: right,
          y2: numberLine ? 112 : bottom,
          class: "mechanism-axis",
        }),
      );
      if (!numberLine)
        drawing.append(
          svg("line", {
            x1: left,
            y1: top,
            x2: left,
            y2: bottom,
            class: "mechanism-axis",
          }),
        );
      const tickEnds = [];
      let tickBottom = 0;
      for (const tick of (numberLine ? o.config.ticks : o.config.x.ticks) || [
        xr.min,
        xr.max,
      ]) {
        drawing.append(
          svg("line", {
            x1: X(tick),
            x2: X(tick),
            y1: (numberLine ? 112 : bottom) - 4,
            y2: (numberLine ? 112 : bottom) + 5,
            class: "mechanism-axis",
          }),
        );
        const text = valueText(tick),
          labelWidth = Math.max(12, text.length * 7),
          labelLeft = Math.max(
            0,
            Math.min(
              width - labelWidth,
              (X(tick) / 400) * width - labelWidth / 2,
            ),
          );
        let lane = tickEnds.findIndex((end) => labelLeft > end + 4);
        if (lane < 0) lane = tickEnds.length;
        tickEnds[lane] = labelLeft + labelWidth;
        const label = node("small", "mechanism-tick", text);
        label.style.left = `${labelLeft}px`;
        const y = ((numberLine ? 112 : bottom) / 220) * height + 7 + lane * 15;
        label.style.top = `${y}px`;
        box.append(label);
        tickBottom = Math.max(tickBottom, y + 13);
      }
      box.style.marginBottom = `${Math.max(3, tickBottom - height + 3)}px`;
      const axisCaption = node(
          "p",
          "mechanism-axis-caption",
          numberLine ? o.config.axisLabel : o.config.x.label,
        ),
        seriesKey = node("div", "mechanism-series-key");
      if (numberLine) {
        for (const level of s.levels)
          drawing.append(
            svg("line", {
              x1: X(level),
              x2: X(level),
              y1: 75,
              y2: 150,
              class: "mechanism-level",
            }),
          );
      } else {
        for (const tick of o.config.y.ticks || [yr.min, yr.max]) {
          const label = node("small", "mechanism-y-tick", valueText(tick));
          label.style.top = `${(Y(tick) / 220) * 100}%`;
          label.style.left = `${left / 4 - 1}%`;
          box.append(label);
        }
        axisText(box, o.config.y.label, "mechanism-y-label");
        const patterns = ["", "7 3", "2 3", "10 3 2 3"];
        for (const [i, series] of s.series.entries()) {
          const dash =
            series.role === "reference"
              ? patterns[1 + (i % 3)]
              : patterns[i % 4];
          const p = svg("polyline", {
            points: series.points.map(([x, y]) => `${X(x)},${Y(y)}`).join(" "),
            class: "mechanism-series",
            fill: "none",
            "stroke-dasharray": dash,
          });
          p.dataset.seriesId = series.id;
          p.dataset.role = series.role;
          drawing.append(p);
          const legend = node("span", "mechanism-series-label");
          legend.dataset.role = series.role;
          const swatch = svg("svg", {
            viewBox: "0 0 28 8",
            width: 28,
            height: 8,
            "aria-hidden": "true",
          });
          swatch.append(
            svg("line", {
              x1: 0,
              y1: 4,
              x2: 28,
              y2: 4,
              class: "mechanism-series",
              "stroke-dasharray": dash,
            }),
          );
          legend.append(swatch, node("span", "", series.label));
          seriesKey.append(legend);
        }
      }
      const markerKey = node("ol", "mechanism-marker-key");
      for (const [i, m] of s.markers.entries()) {
        const marker = node("div", "mechanism-marker");
        marker.dataset.markerId = m.id;
        marker.dataset.value = String(numberLine ? m.value : m.y);
        marker.dataset.role = m.role || "output";
        marker.style.left = `${X(numberLine ? m.value : m.x) / 4}%`;
        marker.style.top = `${Y(numberLine ? 0 : m.y) / 2.2}%`;
        const badge = node("span", "mechanism-marker-label", String(i + 1));
        badge.style.bottom = `${15 + i * 17}px`;
        if (Y(numberLine ? 0 : m.y) < 55) {
          badge.style.bottom = "auto";
          badge.style.top = `${15 + i * 17}px`;
        }
        marker.append(node("i", "mechanism-marker-dot"), badge);
        marker.setAttribute(
          "aria-label",
          `${m.label}: ${numberLine ? m.value : `${m.x}, ${m.y}`}`,
        );
        box.append(marker);
        const item = node("li");
        item.dataset.role = m.role || "output";
        item.append(
          node("span", "", `${m.label}: `),
          node(
            "strong",
            "",
            numberLine
              ? valueText(m.value)
              : `${valueText(m.x)}, ${valueText(m.y)}`,
          ),
        );
        markerKey.append(item);
      }
      box.afterAxisCaption = axisCaption;
      box.afterSeriesKey = seriesKey;
      box.afterMarkerKey = markerKey;
      return box;
    }
    function renderObject(o, s) {
      const section = node("section", "mechanism-object");
      section.dataset.objectId = o.id;
      section.dataset.kind = o.kind;
      section.dataset.role = o.role;
      const l = o.layout;
      for (const size of ["wide", "narrow"]) {
        section.style.setProperty(`--${size}-row`, l[size].row + 1);
        section.style.setProperty(`--${size}-col`, l[size].column + 1);
        section.style.setProperty(`--${size}-span`, l[size].columnSpan || 1);
      }
      const objectTitle = node("p", "mechanism-object-title", o.label);
      objectTitle.setAttribute("role", "heading");
      objectTitle.setAttribute("aria-level", "4");
      section.append(objectTitle);
      const body = node("div", "mechanism-object-body");
      section.append(body);
      if (["matrix", "sequence", "pool", "chamber"].includes(o.kind)) {
        let columns =
          o.kind === "matrix"
            ? o.config.columns
            : o.kind === "pool"
              ? o.config.columns || Math.min(o.config.slots, 4)
              : Math.min(o.config.slots, 6);
        const cells = node("div", "mechanism-cells");
        cells.style.setProperty("--columns", columns);
        if (["sequence", "chamber"].includes(o.kind))
          cells.dataset.adaptive = "true";
        body.append(cells);
        const values =
          o.kind === "pool" ? s.slots.map((x) => x?.value ?? null) : s.values;
        values.forEach((v, i) => {
          let label = o.config.slotLabels?.[i];
          if (o.kind === "matrix")
            label = `${o.config.rowLabels?.[Math.floor(i / o.config.columns)] ?? Math.floor(i / o.config.columns)},${o.config.columnLabels?.[i % o.config.columns] ?? i % o.config.columns}`;
          else if (o.kind !== "chamber" || o.config.slotLabels)
            label = label ?? String(i);
          const c = cell(v, i, label);
          if (o.kind === "pool") {
            const slot = s.slots[i];
            c.dataset.owner = slot?.owner || "";
            c.dataset.slotState = slot?.status || "free";
            c.append(node("small", "mechanism-owner", slot?.owner || "Free"));
            if (slot)
              c.append(node("small", "mechanism-slot-state", slot.status));
            else
              c.querySelector(".mechanism-value").setAttribute(
                "aria-label",
                "Free slot",
              );
          }
          const receive = receivingPlans.get(o.id)?.get(i);
          if (receive) {
            const preview = node("small", "mechanism-transfer-preview");
            preview.dataset.receiveIndex = String(i);
            preview.dataset.reserveLength = String(receive.length);
            preview.dataset.formula = String(receive.formula);
            const token = currentSnapshot().tokens.find(
              (t) =>
                t.at.object === o.id &&
                t.at.index === i &&
                t.state === "inflight" &&
                t.label.includes("→"),
            );
            if (token) preview.textContent = token.label;
            c.append(preview);
          }
          cells.append(c);
        });
        if (o.kind === "matrix" && s.window) {
          const w = node("div", "mechanism-window");
          w.dataset.windowId = o.id;
          w.style.left = `${(s.window.column / o.config.columns) * 100}%`;
          w.style.top = `${(s.window.row / o.config.rows) * 100}%`;
          w.style.width = `${(s.window.columns / o.config.columns) * 100}%`;
          w.style.height = `${(s.window.rows / o.config.rows) * 100}%`;
          w.setAttribute(
            "aria-label",
            `Window row ${s.window.row}, column ${s.window.column}`,
          );
          cells.append(w);
        }
      } else if (o.kind === "register") {
        const registerLayout = node("div", "mechanism-register-layout"),
          face = node("div", "mechanism-retained-face");
        face.append(
          node(
            "small",
            "mechanism-committed-label",
            s.value === null ? "Not yet written" : "Committed value",
          ),
        );
        const registerValue = leaf(
          s.value,
          "mechanism-value mechanism-register-value",
        );
        if (typeof s.value === "string") registerValue.dataset.format = "text";
        face.append(registerValue);
        if (s.previous !== undefined || s.expression) {
          const history = node("details", "mechanism-register-history");
          history.append(
            node("summary", "", "Calculation"),
            node("p", "", s.expression || `Previous: ${valueText(s.previous)}`),
          );
          face.append(history);
        }
        registerLayout.append(face);
        const plan = incomingPlans.get(o.id);
        if (plan) {
          const incoming = node("div", "mechanism-incoming-region");
          incoming.setAttribute(
            "aria-label",
            "Incoming values, not yet committed",
          );
          incoming.append(
            node("small", "mechanism-incoming-label", "Incoming"),
          );
          const docks = node("div", "mechanism-incoming-grid");
          for (let j = 0; j < plan.capacity; j++) {
            const dock = node("div", "mechanism-incoming-cell");
            dock.dataset.incomingIndex = String(j);
            docks.append(dock);
          }
          incoming.append(docks);
          registerLayout.append(incoming);
        }
        body.append(
          registerLayout,
          node(
            "small",
            "mechanism-register-type",
            [o.config.type, o.config.unit].filter(Boolean).join(" · "),
          ),
        );
      } else if (o.kind === "numberLine" || o.kind === "plot") {
        const plot = chart(o, s, o.kind === "numberLine");
        body.append(
          plot,
          plot.afterAxisCaption,
          plot.afterSeriesKey,
          plot.afterMarkerKey,
        );
      } else if (o.kind === "lanes") {
        const lanes = node("div", "mechanism-lanes");
        body.append(lanes);
        o.config.lanes.forEach((label, i) => {
          const row = node("div", "mechanism-lane");
          row.dataset.laneIndex = String(i);
          row.append(node("span", "mechanism-lane-label", label));
          const track = node("div", "mechanism-lane-track");
          for (const task of s.segments.filter((x) => x.lane === i)) {
            const bar = node("div", "mechanism-segment");
            bar.dataset.segmentId = task.id;
            bar.dataset.fullLabel = task.label;
            bar.dataset.shortLabel =
              "T" + segmentOrder.get(o.id + "|" + task.id);
            bar.dataset.state = task.state;
            bar.dataset.role = task.role;
            bar.style.left = `${((task.start - o.config.start) / (o.config.end - o.config.start)) * 100}%`;
            bar.style.width = `${((task.end - task.start) / (o.config.end - o.config.start)) * 100}%`;
            bar.append(
              node("span", "", task.label),
              node("small", "", task.state),
            );
            bar.setAttribute(
              "aria-label",
              `${task.label} · ${task.state} · ${task.start}–${task.end}`,
            );
            track.append(bar);
          }
          if (s.cursor !== undefined && s.cursor !== null) {
            const cursor = node("i", "mechanism-lane-cursor");
            cursor.style.left = `${((s.cursor - o.config.start) / (o.config.end - o.config.start)) * 100}%`;
            track.append(cursor);
          }
          row.append(track);
          lanes.append(row);
        });
        const ticks = node("div", "mechanism-lane-ticks");
        for (const tick of o.config.ticks || [o.config.start, o.config.end]) {
          const t = node("small", "", tick);
          t.dataset.axisPosition = String(
            (tick - o.config.start) / (o.config.end - o.config.start),
          );
          t.style.left = `${Number(t.dataset.axisPosition) * 100}%`;
          ticks.append(t);
        }
        body.append(
          ticks,
          node("small", "mechanism-lane-axis", o.config.axisLabel),
        );
        const taskKey = node("ul", "mechanism-task-key");
        for (const task of s.segments) {
          const item = node(
            "li",
            "",
            `T${segmentOrder.get(o.id + "|" + task.id)}: ${task.label} · ${o.config.lanes[task.lane]} · ${valueText(task.start)}–${valueText(task.end)} · ${task.state}`,
          );
          item.dataset.taskKeyId = task.id;
          taskKey.append(item);
        }
        taskKey.hidden = true;
        body.append(taskKey);
      }
      if (s.status)
        section.append(node("p", "mechanism-object-status", s.status));
      return section;
    }
    function anchorRect(a, receiving = false) {
      if (!a) return null;
      const object = objectNode(a.object);
      if (!object) return null;
      if (a.index !== undefined && objects.get(a.object)?.kind === "register")
        return (
          object
            .querySelector(`[data-incoming-index="${a.index}"]`)
            ?.getBoundingClientRect() || null
        );
      if (a.index !== undefined) {
        const target =
          object.querySelector(`[data-cell-index="${a.index}"]`) ||
          object.querySelector(
            `[data-lane-index="${a.index}"] .mechanism-lane-track`,
          );
        return (
          target?.querySelector("[data-value]")?.getBoundingClientRect() ||
          target?.getBoundingClientRect() ||
          object.getBoundingClientRect()
        );
      }
      if (a.x !== undefined) {
        const plot = object.querySelector(
            ".mechanism-number-line,.mechanism-plot",
          ),
          r = plot.getBoundingClientRect(),
          g = plot.mechanismGeometry;
        const x = r.left + (g.X(a.x) / 400) * r.width,
          y = r.top + (g.Y(a.y || 0) / 220) * r.height;
        return { left: x - 12, top: y - 12, width: 24, height: 24 };
      }
      if (objects.get(a.object)?.kind === "register")
        return object
          .querySelector(".mechanism-register-value")
          .getBoundingClientRect();
      return (
        object
          .querySelector(".mechanism-object-body")
          ?.getBoundingClientRect() || object.getBoundingClientRect()
      );
    }
    function drawConnections(snapshot) {
      scene
        .querySelectorAll(".mechanism-connections,.mechanism-connection-notes")
        .forEach((n) => n.remove());
      const active = (lesson.connections || []).filter((c) =>
        snapshot.activeConnections.includes(c.id),
      );
      if (!active.length) return;
      const notes = node("div", "mechanism-connection-notes");
      for (const c of active) notes.append(node("span", "", c.label));
      scene.append(notes);
      const drawing = svg("svg", {
          class: "mechanism-connections",
          "aria-hidden": "true",
        }),
        base = scene.getBoundingClientRect(),
        id = "mechanism-arrow-" + ++serial;
      drawing.setAttribute(
        "viewBox",
        `0 0 ${Math.max(1, base.width)} ${Math.max(1, base.height)}`,
      );
      drawing.setAttribute("preserveAspectRatio", "none");
      const defs = svg("defs"),
        marker = svg("marker", {
          id,
          viewBox: "0 0 8 8",
          refX: 7,
          refY: 4,
          markerWidth: 6,
          markerHeight: 6,
          orient: "auto-start-reverse",
        });
      marker.append(svg("path", { d: "M0 0 L8 4 L0 8 Z", fill: "#7e317b" }));
      defs.append(marker);
      drawing.append(defs);
      for (const c of active) {
        const a = anchorRect(c.from),
          b = anchorRect(c.to);
        if (!a || !b) continue;
        const ax = a.left + a.width / 2,
          ay = a.top + a.height / 2,
          bx = b.left + b.width / 2,
          by = b.top + b.height / 2,
          dx = bx - ax,
          dy = by - ay;
        if (!dx && !dy) continue;
        const edge = (r) =>
          1 /
          Math.max(
            Math.abs(dx) / Math.max(1, r.width / 2),
            Math.abs(dy) / Math.max(1, r.height / 2),
          );
        const u = Math.min(0.48, edge(a)),
          v = Math.min(0.48, edge(b));
        const line = svg("line", {
          x1: ax + dx * u - base.left,
          y1: ay + dy * u - base.top,
          x2: bx - dx * v - base.left,
          y2: by - dy * v - base.top,
          class: "mechanism-connection",
        });
        line.dataset.connectionId = c.id;
        if (c.kind !== "association")
          line.setAttribute("marker-end", `url(#${id})`);
        drawing.append(line);
      }
      scene.prepend(drawing);
    }
    function positionTokens(snapshot) {
      for (const el of all("[data-token-id]")) {
        if (el.classList.contains("mechanism-token-integrated")) {
          if (el.dataset.partial === "true") {
            const parent = el.parentElement;
            parent.replaceChildren(
              document.createTextNode(
                parent.dataset.numeric === "true"
                  ? valueText(Number(parent.dataset.value))
                  : parent.dataset.value,
              ),
            );
          } else el.replaceWith(...el.childNodes);
        } else el.remove();
      }
      grid.querySelectorAll(".mechanism-token-rail").forEach((n) => n.remove());
      for (const token of snapshot.tokens) {
        const object = objectNode(token.at.object);
        if (!object) continue;
        const cell =
          token.at.index === undefined
            ? null
            : object.querySelector(`[data-cell-index="${token.at.index}"]`);
        const baseline =
          cell?.querySelector("[data-value]") ||
          (!cell && objects.get(token.at.object)?.kind === "register"
            ? object.querySelector(".mechanism-register-value")
            : null);
        const el = node("span", "mechanism-token-integrated");
        el.dataset.tokenId = token.id;
        el.dataset.role = token.role;
        el.dataset.owner = token.owner;
        el.dataset.tokenState = token.state;
        el.title = `${token.label}${token.owner ? " · " + token.owner : ""}${token.state === "copy" ? " · copy/read" : ""}`;
        el.setAttribute("aria-label", el.title);
        if (
          baseline &&
          baseline.dataset.value === String(token.value) &&
          !baseline.closest("[data-token-id]")
        ) {
          baseline.replaceWith(el);
          el.append(baseline);
          continue;
        }
        const display = valueText(token.value),
          start = baseline?.textContent.indexOf(display) ?? -1;
        if (
          baseline &&
          start >= 0 &&
          display &&
          display !== "—" &&
          !baseline.querySelector("[data-token-id]")
        ) {
          const text = baseline.textContent;
          el.dataset.partial = "true";
          el.append(leaf(token.value));
          baseline.replaceChildren(
            document.createTextNode(text.slice(0, start)),
            el,
            document.createTextNode(text.slice(start + display.length)),
          );
          continue;
        }
        // A token carrying a subset/different representation is still shown,
        // but never pasted over a resident value or its ownership metadata.
        let rail = object.querySelector(".mechanism-token-rail");
        if (!rail) {
          rail = node("div", "mechanism-token-rail");
          object.querySelector(".mechanism-object-body").append(rail);
        }
        el.className = "mechanism-token-caption";
        el.append(
          node(
            "small",
            "",
            `${token.state === "copy" ? "Read copy" : "Data"}${token.at.index !== undefined ? " · " + token.at.index : ""}: `,
          ),
          leaf(token.value),
        );
        rail.append(el);
      }
    }
    function stickyTop() {
      stickyFrame = 0;
      let top = 0;
      if (!expanded)
        for (const el of document.querySelectorAll(
          "header,nav,.fixed-top,#quarto-announcement",
        )) {
          const style = getComputedStyle(el),
            r = el.getBoundingClientRect();
          if (
            ["fixed", "sticky"].includes(style.position) &&
            r.top <= 2 &&
            r.bottom > 0 &&
            r.width > innerWidth * 0.5
          )
            top = Math.max(top, r.bottom);
        }
      component.style.setProperty("--mechanism-sticky-top", `${top + 4}px`);
    }
    function layout() {
      if (destroyed) return;
      stickyTop();
      component.classList.toggle("mechanism-narrow", narrow());
      for (const cells of all("[data-adaptive]")) {
        const values = [
            ...cells.querySelectorAll(
              ".mechanism-cell > .mechanism-value,.mechanism-cell > .mechanism-token-integrated > .mechanism-value",
            ),
          ].map((x) => x.textContent),
          longest = Math.max(1, ...values.map((x) => x.length)),
          minWidth =
            longest > 18 ? 150 : longest > 10 ? 110 : longest > 5 ? 82 : 60;
        const count = cells.querySelectorAll(".mechanism-cell").length;
        cells.style.setProperty(
          "--columns",
          Math.max(
            1,
            Math.min(
              count,
              6,
              Math.floor(
                (cells.getBoundingClientRect().width + 4) / (minWidth + 4),
              ),
            ),
          ),
        );
      }
      for (const o of lesson.objects.filter((o) => o.kind === "pool")) {
        const object = objectNode(o.id),
          body = object.querySelector(".mechanism-object-body"),
          cells = object.querySelector(".mechanism-cells"),
          state = currentSnapshot().objects[o.id],
          columns = o.config.columns || Math.min(o.config.slots, 4);
        object.querySelector(".mechanism-ownership-key")?.remove();
        cells.style.minWidth = "";
        cells.style.setProperty("--columns", columns);
        const simple = state.slots
            .filter(Boolean)
            .every((slot) => String(slot.value).length <= 3),
          dense = body.clientWidth / columns < 45 && simple;
        object.classList.toggle("mechanism-dense-pool", dense);
        if (dense) {
          const key = node("div", "mechanism-ownership-key"),
            seen = new Set();
          for (const slot of state.slots) {
            if (!slot) continue;
            const label = `${slot.value}: ${slot.owner} · ${slot.status}`;
            if (!seen.has(label)) {
              key.append(node("span", "", label));
              seen.add(label);
            }
          }
          if (state.slots.some((x) => x === null))
            key.append(node("span", "", "— = free"));
          object.append(key);
        } else {
          const words = state.slots
            .filter(Boolean)
            .flatMap((slot) =>
              [String(slot.value), slot.owner].flatMap((text) =>
                text.split(/\s+/),
              ),
            );
          const min = Math.max(
            64,
            Math.min(
              110,
              Math.max(1, ...words.map((x) => x.length)) * 7.5 + 10,
            ),
          );
          cells.style.setProperty(
            "--columns",
            Math.max(
              1,
              Math.min(columns, Math.floor((body.clientWidth + 4) / (min + 4))),
            ),
          );
        }
      }
      for (const o of lesson.objects.filter((o) =>
        ["plot", "numberLine"].includes(o.kind),
      )) {
        const body = objectNode(o.id).querySelector(".mechanism-object-body"),
          plot = chart(
            o,
            currentSnapshot().objects[o.id],
            o.kind === "numberLine",
            body.getBoundingClientRect().width,
          );
        body.replaceChildren(
          plot,
          plot.afterAxisCaption,
          plot.afterSeriesKey,
          plot.afterMarkerKey,
        );
      }
      for (const [id, plan] of incomingPlans) {
        const object = objectNode(id),
          layout = object.querySelector(".mechanism-register-layout"),
          width = layout.getBoundingClientRect().width,
          wide =
            width >= 280 &&
            typeof currentSnapshot().objects[id].value !== "string";
        layout.classList.toggle("mechanism-register-wide", wide);
        const grid = object.querySelector(".mechanism-incoming-grid"),
          columns = Math.min(plan.capacity, wide ? 2 : plan.length > 3 ? 2 : 4);
        grid.style.gridTemplateColumns = `repeat(${columns},minmax(0,1fr))`;
        const cellWidth =
          (width / (wide ? 2 : 1) - (columns - 1) * 4) / columns;
        for (const dock of grid.children)
          dock.style.minHeight = `${Math.max(30, Math.ceil((plan.length * 7) / Math.max(20, cellWidth - 6)) * 16 + 8)}px`;
      }
      for (const preview of all(".mechanism-transfer-preview")) {
        const width = preview.getBoundingClientRect().width,
          length = Number(preview.dataset.reserveLength) || 1;
        preview.style.minHeight = `${Math.max(20, Math.ceil((Math.min(120, length) * 6) / Math.max(20, width)) * 14)}px`;
      }
      positionTokens(currentSnapshot());
      drawConnections(currentSnapshot());
      for (const object of all("[data-kind=lanes]")) {
        const key = object.querySelector(".mechanism-task-key");
        let any = false;
        for (const track of object.querySelectorAll(".mechanism-lane-track")) {
          track
            .querySelectorAll(
              ".mechanism-task-callout,.mechanism-callout-leaders",
            )
            .forEach((n) => n.remove());
          const rows = [],
            callouts = [];
          for (const bar of track.querySelectorAll(".mechanism-segment")) {
            const compact =
              bar.clientWidth <
              Math.min(170, bar.dataset.fullLabel.length * 6.5 + 12);
            bar.classList.toggle("mechanism-compact-segment", compact);
            const entry = [...key.children].find(
              (n) => n.dataset.taskKeyId === bar.dataset.segmentId,
            );
            if (entry) entry.hidden = !compact;
            if (!compact) continue;
            any = true;
            const center = bar.offsetLeft + bar.offsetWidth / 2,
              w = bar.dataset.shortLabel.length * 7 + 6,
              left = Math.max(
                0,
                Math.min(track.clientWidth - w, center - w / 2),
              );
            let row = rows.findIndex((end) => left > end + 4);
            if (row < 0) row = rows.length;
            rows[row] = left + w;
            const label = node(
              "span",
              "mechanism-task-callout",
              bar.dataset.shortLabel,
            );
            label.style.left = `${left}px`;
            label.style.top = `${row * 17}px`;
            label.dataset.segmentFor = bar.dataset.segmentId;
            track.append(label);
            callouts.push({
              x: left + w / 2,
              y: row * 17 + 13,
              target: center,
            });
          }
          const extra = rows.length ? rows.length * 17 + 3 : 0;
          track.style.height = `${54 + extra}px`;
          track.style.setProperty("--callout-space", `${extra + 3}px`);
          if (callouts.length) {
            const leaders = svg("svg", {
              class: "mechanism-callout-leaders",
              viewBox: `0 0 ${Math.max(1, track.clientWidth)} ${54 + extra}`,
              "aria-hidden": "true",
            });
            for (const c of callouts)
              leaders.append(
                svg("line", {
                  x1: c.x,
                  y1: c.y,
                  x2: c.target,
                  y2: extra + 3,
                  class: "mechanism-callout-line",
                }),
              );
            track.prepend(leaders);
          }
        }
        const ticks = object.querySelector(".mechanism-lane-ticks"),
          ends = [];
        for (const tick of [...ticks.children].sort(
          (a, b) =>
            Number(a.dataset.axisPosition) - Number(b.dataset.axisPosition),
        )) {
          const width = tick.getBoundingClientRect().width,
            left = Math.max(
              0,
              Math.min(
                ticks.clientWidth - width,
                Number(tick.dataset.axisPosition) * ticks.clientWidth -
                  width / 2,
              ),
            );
          let lane = ends.findIndex((end) => left > end + 4);
          if (lane < 0) lane = ends.length;
          ends[lane] = left + width;
          tick.style.left = `${left}px`;
          tick.style.top = `${lane * 15}px`;
          tick.style.transform = "none";
        }
        ticks.style.height = `${Math.max(1, ends.length) * 15}px`;
        key.hidden = !any;
      }
      lastNarrow = narrow();
    }
    function paint() {
      grid.replaceChildren(
        ...lesson.objects.map((o) =>
          renderObject(o, currentSnapshot().objects[o.id]),
        ),
      );
      overlay.replaceChildren();
      metrics.replaceChildren();
      for (const m of currentSnapshot().metrics) {
        const pair = node("div");
        pair.dataset.metricId = m.id;
        pair.append(node("dt", "", m.label));
        const dd = node("dd");
        dd.append(leaf(m.value));
        if (
          m.unit &&
          !(
            [
              "tokens",
              "positions",
              "values",
              "slots",
              "pages",
              "requests",
              "entries",
              "items",
              "outputs",
              "parameters",
            ].includes(m.unit.toLowerCase()) &&
            m.label
              .toLowerCase()
              .split(/[^a-z]+/)
              .includes(m.unit.toLowerCase())
          )
        )
          dd.append(document.createTextNode(" " + m.unit));
        pair.append(dd);
        metrics.append(pair);
      }
      result.textContent = currentSnapshot().result;
      layout();
      updateDataset();
    }
    function clearHold() {
      if (holdTimer !== null) {
        clearTimeout(holdTimer);
        holdTimer = null;
        holdRemaining = Math.max(
          0,
          holdRemaining - (performance.now() - holdStarted),
        );
      }
    }
    function cleanEffects() {
      for (const a of effects) a.cancel();
      effects = [];
      overlay
        .querySelectorAll(".mechanism-travel,.mechanism-motion-path")
        .forEach((x) => x.remove());
      grid.querySelectorAll("[data-incoming-value]").forEach((n) => {
        n.textContent = "";
        delete n.dataset.incomingValue;
      });
      grid
        .querySelectorAll(".mechanism-receiving")
        .forEach((x) => x.classList.remove("mechanism-receiving"));
      grid.querySelectorAll("[data-motion-hidden]").forEach((x) => {
        x.style.visibility = "";
        delete x.dataset.motionHidden;
      });
    }
    function cancel() {
      epoch++;
      clearHold();
      cleanEffects();
      motionFinished = null;
      moving = false;
    }
    function pause() {
      clearHold();
      autoplay = false;
      paused = true;
      for (const a of effects) if (a.playState === "running") a.pause();
      updateDataset();
    }
    function settle() {
      if (!moving) return;
      const finish = motionFinished;
      finish?.();
    }
    function seek(target) {
      cancel();
      autoplay = false;
      paused = true;
      index = committed = Math.max(0, Math.min(steps.length - 1, target));
      holdRemaining = 0;
      paint();
    }
    function schedule() {
      if (!autoplay || paused || index === steps.length - 1) return;
      holdRemaining = holdRemaining || readingPauseMs;
      holdStarted = performance.now();
      holdTimer = setTimeout(() => {
        holdTimer = null;
        holdRemaining = 0;
        if (autoplay && !paused) advance(true);
      }, holdRemaining);
    }
    function animate(el, frames, duration) {
      const a = el.animate(frames, {
        duration,
        easing: "ease-in-out",
        fill: "forwards",
      });
      effects.push(a);
      return a;
    }
    function prepareMotion(before, after, t) {
      for (const [id] of incomingPlans) {
        if (
          same(before.objects[id].value, after.objects[id].value) &&
          t.to?.object !== id
        )
          continue;
        const object = objectNode(id),
          values = incomingValues(before, after, t, id);
        object.classList.add("mechanism-receiving");
        values.forEach((value, i) => {
          const dock = object.querySelector(`[data-incoming-index="${i}"]`);
          if (!dock) return;
          dock.replaceChildren(
            node("span", "mechanism-pending-value", valueText(value)),
          );
          dock.dataset.incomingValue = value === null ? "null" : String(value);
        });
      }
      for (const [id, map] of receivingPlans) {
        const beforeState = before.objects[id],
          afterState = after.objects[id],
          av =
            beforeState.values ||
            beforeState.slots?.map((x) => x?.value ?? null),
          bv =
            afterState.values || afterState.slots?.map((x) => x?.value ?? null);
        for (const [index] of map) {
          const preview = objectNode(id).querySelector(
            `[data-receive-index="${index}"]`,
          );
          if (!preview) continue;
          const token = after.tokens.find(
            (x) =>
              x.at.object === id &&
              x.at.index === index &&
              !before.tokens.some(
                (old) =>
                  old.id === x.id &&
                  same(old.at, x.at) &&
                  same(old.value, x.value),
              ),
          );
          if (token?.state === "inflight" && token.label.includes("→"))
            preview.textContent = token.label;
          else if (token || (bv && !same(av[index], bv[index])))
            preview.textContent =
              "Incoming " + valueText(token ? token.value : bv[index]);
          preview.dataset.active = "true";
        }
      }
      layout();
    }
    function segmentHits(p, q, r) {
      let lo = 0,
        hi = 1;
      const dx = q.x - p.x,
        dy = q.y - p.y;
      for (const [v, d, min, max] of [
        [p.x, dx, r.left, r.right],
        [p.y, dy, r.top, r.bottom],
      ]) {
        if (Math.abs(d) < 1e-8) {
          if (v < min || v > max) return false;
          continue;
        }
        let a = (min - v) / d,
          b = (max - v) / d;
        if (a > b) [a, b] = [b, a];
        lo = Math.max(lo, a);
        hi = Math.min(hi, b);
        if (lo > hi) return false;
      }
      return true;
    }
    function transferRoute(start, end, from, to, halfWidth, halfHeight) {
      const base = scene.getBoundingClientRect(),
        source = objectNode(from.object).getBoundingClientRect(),
        target = objectNode(to.object).getBoundingClientRect();
      const contains = (r, p) =>
        p.x >= r.left && p.x <= r.right && p.y >= r.top && p.y <= r.bottom;
      const obstacles = [
        ...scene.querySelectorAll(
          ".mechanism-object-title,.mechanism-object-status,.mechanism-register-type,.mechanism-retained-face,.mechanism-incoming-label,.mechanism-cell-index,.mechanism-owner,.mechanism-slot-state,.mechanism-transfer-preview",
        ),
      ]
        .filter(
          (el) =>
            el.textContent.trim() &&
            getComputedStyle(el).visibility !== "hidden",
        )
        .map((el) => el.getBoundingClientRect())
        .filter(
          (r) =>
            r.width && r.height && !contains(r, start) && !contains(r, end),
        )
        .map((r) => ({
          left: r.left - halfWidth - 4,
          right: r.right + halfWidth + 4,
          top: r.top - halfHeight - 4,
          bottom: r.bottom + halfHeight + 4,
        }));
      const simplify = (points) =>
        points.filter(
          (p, i) =>
            !i ||
            Math.abs(p.x - points[i - 1].x) + Math.abs(p.y - points[i - 1].y) >
              0.1,
        );
      const clear = (points) =>
        points.every(
          (p) =>
            p.x - halfWidth >= base.left - 1 &&
            p.x + halfWidth <= base.right + 1 &&
            p.y - halfHeight >= base.top - 1 &&
            p.y + halfHeight <= base.bottom + 1,
        ) &&
        points
          .slice(1)
          .every((p, i) =>
            obstacles.every((r) => !segmentHits(points[i], p, r)),
          );
      const candidates = [
        [start, end],
        [start, { x: start.x, y: end.y }, end],
        [start, { x: end.x, y: start.y }, end],
      ];
      const rails = [
        base.left + 9,
        base.right - 9,
        source.left - 9,
        source.right + 9,
        target.left - 9,
        target.right + 9,
      ];
      for (const x of rails)
        candidates.push([start, { x, y: start.y }, { x, y: end.y }, end]);
      const ys = [
        source.top - 9,
        source.bottom + 9,
        target.top - 9,
        target.bottom + 9,
      ];
      for (const sx of [source.left - 9, source.right + 9, start.x])
        for (const tx of [target.left - 9, target.right + 9, end.x])
          for (const y of ys)
            candidates.push([
              start,
              { x: sx, y: start.y },
              { x: sx, y },
              { x: tx, y },
              { x: tx, y: end.y },
              end,
            ]);
      const valid = candidates.map(simplify).filter(clear);
      valid.sort((a, b) => {
        const length = (p) =>
          p
            .slice(1)
            .reduce((n, q, i) => n + Math.hypot(q.x - p[i].x, q.y - p[i].y), 0);
        return length(a) - length(b);
      });
      return valid[0] || null;
    }
    function travel(
      label,
      from,
      to,
      duration,
      role = "input",
      rawValue = label,
    ) {
      if (
        objects.get(to.object)?.kind === "register" &&
        to.index === undefined &&
        objectNode(to.object).classList.contains("mechanism-receiving")
      )
        to = { ...to, index: 0 };
      const a = anchorRect(from),
        b = anchorRect(to, true);
      if (!a || !b) return false;
      const start = { x: a.left + a.width / 2, y: a.top + a.height / 2 },
        end = { x: b.left + b.width / 2, y: b.top + b.height / 2 },
        base = scene.getBoundingClientRect();
      const chip = node("span", "mechanism-travel", label);
      chip.dataset.role = role;
      chip.dataset.motionValue = String(rawValue);
      chip.dataset.motionLabel = label;
      if (from.index !== undefined)
        chip.dataset.sourceIndex = String(from.index);
      if (to.index !== undefined) chip.dataset.targetIndex = String(to.index);
      chip.dataset.sourceObject = from.object;
      chip.dataset.targetObject = to.object;
      chip.style.left = `${start.x - base.left}px`;
      chip.style.top = `${start.y - base.top}px`;
      overlay.append(chip);
      const size = chip.getBoundingClientRect();
      let path = transferRoute(
          start,
          end,
          from,
          to,
          size.width / 2,
          size.height / 2,
        ),
        carrier = false;
      if (!path) {
        carrier = true;
        chip.classList.add("mechanism-carrier");
        chip.textContent = "";
        path = transferRoute(start, end, from, to, 4, 4);
      }
      if (!path) {
        chip.remove();
        return false;
      }
      chip.dataset.motionMode = carrier ? "carrier" : "value";
      const drawing = svg("svg", {
        class: "mechanism-motion-path",
        viewBox: `0 0 ${Math.max(1, base.width)} ${Math.max(1, base.height)}`,
        "aria-hidden": "true",
      });
      drawing.append(
        svg("polyline", {
          points: path
            .map((p) => `${p.x - base.left},${p.y - base.top}`)
            .join(" "),
          class: "mechanism-transfer-track",
          fill: "none",
        }),
      );
      overlay.prepend(drawing);
      const lengths = [0];
      for (let i = 1; i < path.length; i++)
        lengths.push(
          lengths[i - 1] +
            Math.hypot(path[i].x - path[i - 1].x, path[i].y - path[i - 1].y),
        );
      const total = lengths.at(-1) || 1;
      animate(
        chip,
        path.map((p, i) => ({
          transform: `translate(calc(-50% + ${p.x - start.x}px),calc(-50% + ${p.y - start.y}px))`,
          offset: lengths[i] / total,
          opacity: 1,
        })),
        duration,
      );
      return true;
    }
    function illustrate(before, after, t, duration) {
      let count = 0;
      for (const token of after.tokens) {
        const old = before.tokens.find((x) => x.id === token.id);
        if (
          old &&
          same(old.at, token.at) &&
          same(old.value, token.value) &&
          same(old.sourceAt, token.sourceAt) &&
          old.sourceId === token.sourceId
        )
          continue;
        const source =
          old && !same(old.at, token.at)
            ? old.at
            : token.sourceAt ||
              (token.sourceId
                ? before.tokens.find((x) => x.id === token.sourceId)?.at
                : null);
        if (
          source &&
          (token.state === "copy" || !old || !same(old.at, token.at))
        ) {
          if (
            travel(
              valueText(token.value),
              source,
              token.at,
              duration,
              token.role,
              token.value,
            )
          )
            count++;
        }
      }
      if (t.kind === "window")
        for (const o of lesson.objects.filter((x) => x.kind === "matrix")) {
          const a = before.objects[o.id].window,
            b = after.objects[o.id].window;
          if (!b || same(a, b)) continue;
          if (!a) {
            const cells = objectNode(o.id)?.querySelector(".mechanism-cells");
            if (cells) {
              const w = node("div", "mechanism-window");
              w.style.left = `${(b.column / o.config.columns) * 100}%`;
              w.style.top = `${(b.row / o.config.rows) * 100}%`;
              w.style.width = `${(b.columns / o.config.columns) * 100}%`;
              w.style.height = `${(b.rows / o.config.rows) * 100}%`;
              cells.append(w);
              animate(
                w,
                [
                  { opacity: 0, transform: "scale(.75)" },
                  { opacity: 1, transform: "scale(1)" },
                ],
                duration,
              );
              count++;
            }
            continue;
          }
          const w = objectNode(o.id)?.querySelector(".mechanism-window"),
            cells = objectNode(o.id)?.querySelector(".mechanism-cells");
          if (w && cells) {
            const r = cells.getBoundingClientRect();
            animate(
              w,
              [
                { transform: "translate(0,0)" },
                {
                  transform: `translate(${((b.column - a.column) / o.config.columns) * r.width}px,${((b.row - a.row) / o.config.rows) * r.height}px)`,
                },
              ],
              duration,
            );
            count++;
          }
        }
      if (t.kind === "release") {
        for (const o of lesson.objects) {
          const a = before.objects[o.id],
            b = after.objects[o.id];
          if (o.kind === "pool")
            a.slots.forEach((slot, i) => {
              if (slot && !b.slots[i]) {
                const c = objectNode(o.id).querySelector(
                  `[data-cell-index="${i}"] .mechanism-value`,
                );
                animate(
                  c,
                  [
                    { opacity: 1, transform: "scale(1)" },
                    { opacity: 0, transform: "scale(.65)" },
                  ],
                  duration,
                );
                count++;
              }
            });
        }
        for (const token of before.tokens)
          if (!after.tokens.some((x) => x.id === token.id)) {
            const el = all("[data-token-id]").find(
              (x) => x.dataset.tokenId === token.id,
            );
            if (el) {
              animate(
                el.querySelector("[data-value]") || el,
                [{ opacity: 1 }, { opacity: 0 }],
                duration,
              );
              count++;
            }
          }
      }
      // Position markers exactly at declared endpoints. Physical plot curves
      // never get an invented straight-line trajectory between samples.
      for (const o of lesson.objects.filter((x) =>
        ["numberLine", "plot"].includes(x.kind),
      )) {
        const a = before.objects[o.id],
          b = after.objects[o.id];
        for (const marker of b.markers) {
          const previous = a.markers.find((x) => x.id === marker.id);
          if (previous && same(previous, marker)) continue;
          const el = objectNode(o.id)?.querySelector(
            `[data-marker-id="${CSS.escape(marker.id)}"]`,
          );
          if (el) {
            if (
              o.kind === "numberLine" &&
              previous &&
              lesson.family === "representation" &&
              t.kind === "transform"
            ) {
              const from = anchorRect({ object: o.id, x: previous.value }),
                to = anchorRect({ object: o.id, x: marker.value });
              animate(
                el,
                [
                  { transform: "translate(0,0)" },
                  { transform: `translate(${to.left - from.left}px,0)` },
                ],
                duration,
              );
            } else
              animate(
                el,
                [{ opacity: 1 }, { opacity: 0.15 }, { opacity: 1 }],
                duration,
              );
            count++;
          }
        }
        if (o.kind === "plot" && !same(a.series, b.series)) {
          const drawing = objectNode(o.id).querySelector("svg");
          for (const line of drawing.querySelectorAll("polyline")) {
            animate(line, [{ opacity: 1 }, { opacity: 0.2 }], duration);
            count++;
          }
        }
      }
      if (t.kind === "execute" && t.from && t.to && count === 0) {
        const source = before.objects[t.from.object],
          values =
            source?.values || source?.slots?.map((x) => x?.value ?? null);
        if (values && values.length <= 8) {
          values.forEach((v, i) => {
            if (
              v !== null &&
              travel(
                valueText(v),
                { object: t.from.object, index: i },
                objects.get(t.to.object)?.kind === "register"
                  ? { ...t.to, index: i }
                  : t.to,
                duration,
                objects.get(t.from.object).role,
                v,
              )
            )
              count++;
          });
        }
      }
      if (t.from && t.to && count === 0) {
        const destination = after.objects[t.to.object],
          source = before.objects[t.from.object],
          values =
            destination.slots?.map((x) => x?.value ?? null) ||
            destination.values;
        let label = t.subjects
          .map((id) => after.tokens.find((x) => x.id === id)?.label)
          .filter(Boolean)
          .join(" · ");
        if (!label)
          label =
            t.to.index !== undefined
              ? valueText(
                  values?.[t.to.index] ??
                    destination.value ??
                    source?.value ??
                    t.kind,
                )
              : valueText(
                  destination.value ??
                    source?.value ??
                    values
                      ?.filter((x) => x !== null)
                      .slice(0, 4)
                      .join(" · ") ??
                    t.kind,
                );
        count += travel(String(label), t.from, t.to, duration) ? 1 : 0;
      }
      if (!count) {
        for (const o of lesson.objects) {
          const a = before.objects[o.id],
            b = after.objects[o.id];
          if (same(a, b)) continue;
          const section = objectNode(o.id);
          if (o.kind === "lanes") {
            for (const segment of b.segments) {
              const old = a.segments.find((x) => x.id === segment.id);
              if (!old || same(old, segment)) continue;
              const bar = section.querySelector(
                `[data-segment-id="${CSS.escape(segment.id)}"]`,
              );
              if (bar) {
                const sweep = node("i", "mechanism-work-sweep");
                bar.append(sweep);
                animate(
                  sweep,
                  [
                    { transform: "translateX(-105%)" },
                    { transform: "translateX(105%)" },
                  ],
                  duration,
                );
                count++;
              }
            }
          }
          if (["matrix", "sequence", "chamber", "pool"].includes(o.kind)) {
            const av =
                o.kind === "pool"
                  ? a.slots.map((s) => s?.value ?? null)
                  : a.values,
              bv =
                o.kind === "pool"
                  ? b.slots.map((s) => s?.value ?? null)
                  : b.values;
            bv.forEach((v, i) => {
              if (same(av[i], v)) return;
              const c = section.querySelector(`[data-cell-index="${i}"]`),
                preview = c.querySelector(".mechanism-transfer-preview");
              if (preview?.textContent) {
                animate(preview, [{ opacity: 0.2 }, { opacity: 1 }], duration);
                count++;
                return;
              }
              const chip = node(
                "span",
                "mechanism-inplace-result",
                typeof v === "number"
                  ? "→ " + valueText(v)
                  : v === null
                    ? ""
                    : "",
              );
              if (typeof v !== "number")
                chip.classList.add("mechanism-inplace-carrier");
              chip.title =
                v === null
                  ? "Release after this action"
                  : `Incoming value: ${valueText(v)}`;
              c.append(chip);
              animate(
                chip,
                [
                  { opacity: 0, transform: "translateY(-18px) scale(.85)" },
                  { opacity: 1, transform: "translateY(0) scale(1)" },
                ],
                duration,
              );
              count++;
            });
          }
          if (o.kind === "register" && !same(a.value, b.value)) {
            const dock = section.querySelector(".mechanism-incoming-cell");
            if (dock)
              animate(dock, [{ opacity: 0.2 }, { opacity: 1 }], duration);
            count++;
          }
        }
      }
    }
    function advance(continuous = false) {
      if (moving) {
        autoplay = false;
        paused = true;
        settle();
        return;
      }
      if (index >= steps.length - 1) return;
      clearHold();
      const before = currentSnapshot();
      index++;
      moving = true;
      paused = false;
      autoplay = continuous;
      const after = steps[index].snapshot,
        t = steps[index].transition,
        duration = media.matches ? 0 : movementMs,
        version = ++epoch;
      updateDataset();
      result.textContent =
        "Action in progress. The stored result commits when this action finishes.";
      const finish = () => {
        if (destroyed || version !== epoch || !moving) return;
        cleanEffects();
        moving = false;
        committed = index;
        motionFinished = null;
        paused = !autoplay;
        paint();
        if (autoplay) {
          if (index === steps.length - 1) {
            autoplay = false;
            paused = true;
            updateDataset();
          } else schedule();
        }
      };
      motionFinished = finish;
      if (!duration) {
        finish();
        return;
      }
      prepareMotion(before, after, t);
      illustrate(before, after, t, duration);
      // This one inert effect is the transition clock; pausing freezes it along
      // with every illustrative effect, without queued sub-actions.
      const clock = animate(
        overlay,
        [{ opacity: 1 }, { opacity: 1 }],
        duration,
      );
      clock.finished.then(finish).catch(() => {});
    }
    function play() {
      if (!paused && (moving || autoplay)) {
        pause();
        return;
      }
      autoplay = true;
      paused = false;
      if (moving) {
        for (const a of effects) if (a.playState === "paused") a.play();
        updateDataset();
        return;
      }
      if (index === steps.length - 1) {
        seek(0);
        autoplay = true;
        paused = false;
      }
      if (holdRemaining) schedule();
      else advance(true);
      updateDataset();
    }
    on(buttons.next, "click", () => advance(false));
    on(buttons.previous, "click", () => seek(index - 1));
    on(buttons.restart, "click", () => seek(0));
    on(buttons.play, "click", play);
    on(sequence, "change", () => seek(Number(sequence.value)));
    on(caseMenu, "change", () => {
      const oldIds = steps.map((x) => x.id),
        target = index;
      cancel();
      autoplay = false;
      paused = true;
      currentCase = cases.find((x) => x.id === caseMenu.value);
      steps = currentCase.steps;
      const matched = same(
        oldIds,
        steps.map((x) => x.id),
      );
      index = committed = matched ? Math.min(target, steps.length - 1) : 0;
      holdRemaining = 0;
      menu();
      paint();
      if (!matched)
        result.textContent =
          "Case changed; sequence restarted. " + result.textContent;
    });
    on(component, "keydown", (event) => {
      if (
        event.defaultPrevented ||
        event.repeat ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey ||
        event.target.closest(
          'input,textarea,select,[contenteditable]:not([contenteditable="false"])',
        )
      )
        return;
      const key = event.key.toLowerCase();
      if (!["n", "p", "k", "r"].includes(key)) return;
      event.preventDefault();
      if (key === "n") advance(false);
      else if (key === "p") seek(index - 1);
      else if (key === "r") seek(0);
      else play();
    });
    function closeWorkspace() {
      if (!expanded) return;
      pause();
      settle();
      dialog.close();
      placeholder.replaceWith(component);
      expanded = false;
      buttons.expand.textContent = "↗";
      buttons.expand.title = "Expand workspace";
      layout();
      buttons.expand.focus({ preventScroll: true });
    }
    on(buttons.expand, "click", () => {
      if (expanded) {
        closeWorkspace();
        return;
      }
      pause();
      settle();
      component.replaceWith(placeholder);
      dialog.append(component);
      expanded = true;
      buttons.expand.textContent = "↙";
      buttons.expand.title = "Return to chapter";
      dialog.showModal();
      layout();
      (buttons.next.disabled ? buttons.previous : buttons.next).focus({
        preventScroll: true,
      });
    });
    on(closeButton, "click", closeWorkspace);
    on(dialog, "cancel", (event) => {
      event.preventDefault();
      closeWorkspace();
    });
    on(dialog, "keydown", (event) => {
      if (event.key !== "Tab") return;
      const focusable = [
        ...dialog.querySelectorAll(
          'button:not([disabled]),select:not([disabled]),input:not([disabled]),textarea:not([disabled]),a[href],summary,[tabindex]:not([tabindex="-1"])',
        ),
      ].filter(
        (el) =>
          el.getClientRects().length &&
          !el.closest("[hidden]") &&
          getComputedStyle(el).visibility === "visible",
      );
      if (!focusable.length) return;
      const active = document.activeElement,
        first = focusable[0],
        last = focusable[focusable.length - 1];
      if (event.shiftKey && (active === first || !focusable.includes(active))) {
        event.preventDefault();
        last.focus();
      } else if (
        !event.shiftKey &&
        (active === last || !focusable.includes(active))
      ) {
        event.preventDefault();
        first.focus();
      }
    });
    on(window, "scroll", () => {
      if (!stickyFrame) stickyFrame = requestAnimationFrame(stickyTop);
    });
    on(document, "transitionend", stickyTop);
    on(document, "visibilitychange", () => {
      if (document.hidden) pause();
    });
    on(context.figure || root, "mlsys-view-change", (event) => {
      if (event.detail.view !== "explore") {
        if (expanded) closeWorkspace();
        pause();
      } else requestAnimationFrame(layout);
    });
    const intersection = new IntersectionObserver((entries) => {
      if (entries.some((e) => !e.isIntersecting) && !expanded) pause();
    });
    intersection.observe(root);
    const resize = new ResizeObserver(() => {
      if (layoutFrame) return;
      layoutFrame = requestAnimationFrame(() => {
        layoutFrame = 0;
        if (!root.isConnected) return;
        const n = narrow();
        if (moving && n !== lastNarrow) {
          pause();
          settle();
        }
        if (!moving) layout();
      });
    });
    resize.observe(component);
    on(media, "change", () => {
      if (media.matches && moving) {
        pause();
        settle();
      }
    });
    menu();
    paint();
    return {
      pause,
      destroy() {
        if (destroyed) return;
        pause();
        cancel();
        if (expanded) {
          dialog.close();
          placeholder.replaceWith(component);
        }
        destroyed = true;
        abort.abort();
        intersection.disconnect();
        resize.disconnect();
        cancelAnimationFrame(layoutFrame);
        cancelAnimationFrame(stickyFrame);
        dialog.remove();
        context.figure?.classList.remove("mlsys-mechanism-figure");
        component.remove();
        root.classList.remove("mlsys-mechanism-panel");
      },
    };
  };
})();
