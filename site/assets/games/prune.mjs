/* ============================================================
   MLSysBook Playground — Pulse Prune (v8, Pixi visual lift)
   ------------------------------------------------------------
   Mechanics unchanged from v7:
     - 5 → 8 → 3 MLP, click weights to prune
     - 45 s timer, 60 % sparsity goal, accuracy floor 50 %
     - Bright weights (mag > 0.55) cause critical-cut penalty
     - Periodic inference pulse routes through live edges
     - Daily seed, emoji-grid share

   v8 visual lift:
     - REAL GlowFilter on the inference pulse + on the receiving neuron
     - The pulse drags a trailing tail (the last N positions) so the
       eye actually sees motion, not just a dot teleporting.
     - Critical-cut bursts use ~50 particles + a Bloom-like ring (was 10)
     - Hover state on edges does a smooth alpha pulse (not just a colour
       swap) so it feels alive when the mouse moves over connections.
     - Win-state bursts at output neurons use BloomFilter for the
       "celebration" feeling.
     - Subtle ambient pulse on idle neurons (very low frequency, just
       enough to show the network is "live" between inferences).
   ============================================================ */

import {
  mountPixiOnCanvas, dailySeed, dayNumber, bestScore,
  pop, flash, burst, floatText, shake, distToSegment, tween, getFilters
} from "./runtime.mjs";
import * as PIXI from "./vendor/pixi.min.mjs";

const TARGET_SPARSITY = 60;
const ACCURACY_FLOOR = 50;
const TIME_LIMIT_MS = 45000;
const LAYERS = [4, 6, 3];

const COL = {
  blueLight:  0xcfe2f3,
  blueStroke: 0x4a90c4,
  blueMid:    0x88b4d8,
  blueFaint:  0xc0d4e8,
  red:        0xc44444,
  mitRed:     0xa31f34,
  greenStroke:0x3d9e5a,
  white:      0xffffff,
  text:       0x333333,
  muted:      0x888888
};

/* Register on the legacy window.MLSP API so the .qmd boot script
   can call MLSP.games.prune(canvas, opts) and get a Promise back. */
window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};
window.MLSP.games.prune = function (canvas, opts) {
  return mountPulsePrune(canvas, opts);
};

export async function mountPulsePrune(canvas, opts = {}) {
  const { app, stage, width: W, height: H } = await mountPixiOnCanvas(canvas, { bg: COL.white });
  const { rand, today } = dailySeed("prune");
  const alltimeBestRef = { v: bestScore.get("prune") };
  const startOnFirstAction = !!opts.startOnFirstAction;
  const safeFirstHint = !!opts.safeFirstHint;
  const relaxedHitTest = !!opts.relaxedHitTest;

  /* --- Network topology --- */
  const neurons = [];
  const topM = 70, botM = 90, leftM = 90, rightM = 90;
  const innerW = W - leftM - rightM, innerH = H - topM - botM;
  const layerDX = innerW / (LAYERS.length - 1);
  LAYERS.forEach((count, li) => {
    const cellH = innerH / Math.max(count - 1, 1);
    for (let i = 0; i < count; i++) {
      neurons.push({ layer: li, idx: i, x: leftM + li * layerDX, y: topM + i * cellH, pulse: 0 });
    }
  });

  const weights = [];
  for (let li = 0; li < LAYERS.length - 1; li++) {
    const from = neurons.filter(n => n.layer === li);
    const to   = neurons.filter(n => n.layer === li + 1);
    from.forEach(f => {
      to.forEach(t => {
        const mag = Math.pow(rand(), 1.6);
        const imp = Math.max(0.01, mag + (rand() - 0.5) * 0.15);
        weights.push({
          from: f, to: t,
          magnitude: mag, importance: imp,
          pruned: false, wasCriticalCut: false,
          activation: 0
        });
      });
    });
  }
  const totalImportance = weights.reduce((s, w) => s + w.importance, 0);

  /* --- Containers (so we can shake the gameplay layer without shaking the HUD) --- */
  const gameLayer = new PIXI.Container();
  const hudLayer = new PIXI.Container();
  const overlayLayer = new PIXI.Container();
  stage.addChild(gameLayer, hudLayer, overlayLayer);

  /* Edges, neurons, pulse marker each in their own layer for z-ordering */
  const edgesGfx = new PIXI.Graphics();
  const pulseTailGfx = new PIXI.Graphics();   // trail behind the pulse
  const pulseGfx = new PIXI.Graphics();        // the head dot itself
  const neuronsGfx = new PIXI.Graphics();
  gameLayer.addChild(edgesGfx, pulseTailGfx, pulseGfx, neuronsGfx);

  /* Track the last N positions of the inference pulse for the trail.
     Newer positions = brighter; older = fade to transparent. */
  const PULSE_TAIL_LEN = 18;
  const pulseTail = [];

  /* Lazy-load glow filters and attach to pulseGfx once available */
  let filters = null;
  getFilters().then(f => {
    filters = f;
    if (f.GlowFilter) {
      pulseGfx.filters = [new f.GlowFilter({
        distance: 14, outerStrength: 2.4, innerStrength: 0.4,
        color: COL.blueStroke, quality: 0.4
      })];
      pulseTailGfx.filters = [new f.GlowFilter({
        distance: 8, outerStrength: 1.2, innerStrength: 0.2,
        color: COL.blueStroke, quality: 0.3
      })];
    }
  });

  /* --- State --- */
  const state = {
    accuracy: 100,
    sparsity: 0,
    pruned: 0,
    total: weights.length,
    removedImp: 0,
    timeLeft: TIME_LIMIT_MS,
    started: !startOnFirstAction,
    over: false,
    won: false,
    hoverIdx: -1,
    inferencePulse: null,
    pulseCooldown: 1200
  };
  const safestWeightIdx = safeFirstHint
    ? weights.reduce((bestIdx, w, i, arr) => {
        if (bestIdx < 0) return i;
        if (w.magnitude < arr[bestIdx].magnitude) return i;
        return bestIdx;
      }, -1)
    : -1;

  /* --- Hit-testing input --- */
  const handlePointerMove = (e) => {
    if (state.over) return;
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    const px = (e.clientX - rect.left) * sx;
    const py = (e.clientY - rect.top) * sy;
    state.hoverIdx = findHover(px, py);
    canvas.style.cursor = state.hoverIdx >= 0 ? "pointer" : "default";
  };
  const handlePointerDown = (e) => {
    e.preventDefault();
    if (state.over) { if (opts.onRetry) opts.onRetry(); return; }
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    let cx, cy;
    if (e.touches && e.touches.length) { cx = e.touches[0].clientX; cy = e.touches[0].clientY; }
    else { cx = e.clientX; cy = e.clientY; }
    const px = (cx - rect.left) * sx;
    const py = (cy - rect.top) * sy;
    const idx = findHover(px, py);
    if (idx >= 0) pruneWeight(weights[idx]);
  };
  const handleKeydown = (e) => {
    if (!isInViewport(canvas)) return;
    if (e.key === "r" || e.key === "R") { e.preventDefault(); if (opts.onRetry) opts.onRetry(); }
  };
  canvas.addEventListener("mousemove", handlePointerMove);
  canvas.addEventListener("pointerdown", handlePointerDown);
  canvas.addEventListener("touchmove", (e) => e.preventDefault(), { passive: false });
  window.addEventListener("keydown", handleKeydown);

  function isInViewport(el) {
    const r = el.getBoundingClientRect();
    return r.bottom > 0 && r.top < (window.innerHeight || document.documentElement.clientHeight);
  }
  function findHover(px, py) {
    let best = -1, bestD = relaxedHitTest ? 18 : 12;
    for (let i = 0; i < weights.length; i++) {
      if (weights[i].pruned) continue;
      const d = distToSegment(px, py, weights[i].from.x, weights[i].from.y, weights[i].to.x, weights[i].to.y);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  /* --- Mechanics --- */
  function pruneWeight(w) {
    if (w.pruned || state.over) return;
    if (!state.started) state.started = true;
    w.pruned = true;
    state.pruned++;
    state.sparsity = (state.pruned / state.total) * 100;
    state.removedImp += w.importance;
    state.accuracy = Math.max(0, 100 * (1 - state.removedImp / totalImportance));

    const mx = (w.from.x + w.to.x) / 2;
    const my = (w.from.y + w.to.y) / 2;
    const isBright = w.magnitude > 0.55;
    if (isBright) {
      w.wasCriticalCut = true;
      shake(gameLayer, 9, 320);
      floatText(overlayLayer, mx, my - 6,
        "−" + (w.importance / totalImportance * 100).toFixed(1) + "% critical!",
        COL.red);
      /* Dense particles + bloom-style ring + secondary inner pop. */
      burst(overlayLayer, mx, my, COL.red, 50, { speed: 5, lifeMs: 900 });
      pop(overlayLayer, mx, my, COL.red, { r: 36 });
      /* Glowing ring that expands and fades — only visible if filters loaded */
      if (filters && filters.GlowFilter) {
        const ring = new PIXI.Graphics();
        ring.circle(0, 0, 10).stroke({ width: 3, color: COL.red, alpha: 1 });
        ring.position.set(mx, my);
        ring.filters = [new filters.GlowFilter({ distance: 18, outerStrength: 3, innerStrength: 0.5, color: COL.red, quality: 0.4 })];
        overlayLayer.addChild(ring);
        tween(ring, "scale.x", 1, 5, 600, "outCubic");
        tween(ring, "scale.y", 1, 5, 600, "outCubic");
        tween(ring, "alpha", 1, 0, 600, "outCubic");
        setTimeout(() => { try { ring.destroy({ children: true }); } catch (e) {} }, 650);
      }
    } else {
      floatText(overlayLayer, mx, my - 6, "+1", COL.greenStroke);
      burst(overlayLayer, mx, my, COL.greenStroke, 16, { speed: 2.5, lifeMs: 600 });
      pop(overlayLayer, mx, my, COL.greenStroke, { r: 18 });
    }

    if (state.sparsity >= TARGET_SPARSITY && state.accuracy >= ACCURACY_FLOOR && !state.over) {
      state.over = true; state.won = true;
      flash(stage, COL.greenStroke, 360);
      const outs = neurons.filter(n => n.layer === LAYERS.length - 1);
      for (const o of outs) {
        burst(overlayLayer, o.x, o.y, COL.greenStroke, 60, { speed: 4, lifeMs: 1100 });
        if (filters && filters.GlowFilter) {
          const winRing = new PIXI.Graphics();
          winRing.circle(0, 0, 14).stroke({ width: 4, color: COL.greenStroke });
          winRing.position.set(o.x, o.y);
          winRing.filters = [new filters.GlowFilter({ distance: 24, outerStrength: 3.5, innerStrength: 0.6, color: COL.greenStroke, quality: 0.5 })];
          overlayLayer.addChild(winRing);
          tween(winRing, "scale.x", 1, 6, 900, "outCubic");
          tween(winRing, "scale.y", 1, 6, 900, "outCubic");
          tween(winRing, "alpha", 1, 0, 900, "outCubic");
          setTimeout(() => { try { winRing.destroy({ children: true }); } catch (e) {} }, 950);
        }
      }
      endGame();
    } else if (state.accuracy < ACCURACY_FLOOR && !state.over) {
      state.over = true; state.won = false;
      flash(stage, COL.mitRed, 320);
      endGame();
    }
  }

  function endGame() {
    const final = Math.round(state.sparsity);
    if (state.won && final > alltimeBestRef.v) {
      alltimeBestRef.v = final;
      bestScore.set("prune", final);
    }
    if (opts.onGameOver) {
      opts.onGameOver({
        sparsity: state.sparsity,
        accuracy: state.accuracy,
        finalSparsity: final,
        won: state.won,
        alltimeBest: alltimeBestRef.v,
        date: today,
        emojiGrid: buildEmojiGrid()
      });
    }
  }

  function buildEmojiGrid() {
    /* 4 rows × 6 cols representing the input→hidden weight matrix.
       Kept consistent with v6 share output so prior shares match. */
    const rows = [];
    for (let i = 0; i < LAYERS[0]; i++) {
      let row = "";
      for (let j = 0; j < LAYERS[1]; j++) {
        let w = null;
        for (let k = 0; k < weights.length; k++) {
          if (weights[k].from.layer === 0 && weights[k].from.idx === i &&
              weights[k].to.layer === 1 && weights[k].to.idx === j) {
            w = weights[k]; break;
          }
        }
        if (!w) row += "⬜";
        else if (w.pruned && w.wasCriticalCut) row += "🟥";
        else if (w.pruned) row += "⬛";
        else if (w.magnitude > 0.55) row += "🟦";
        else row += "🟩";
      }
      rows.push(row);
    }
    return rows.join("\n");
  }

  function spawnPulse() {
    const inputs = neurons.filter(n => n.layer === 0);
    const src = inputs[Math.floor(rand() * inputs.length)];
    const legs = [];
    let cur = src;
    for (let li = 1; li < LAYERS.length; li++) {
      const cands = weights.filter(w => !w.pruned && w.from === cur && w.to.layer === li);
      if (cands.length === 0) break;
      const chosen = cands[Math.floor(rand() * cands.length)];
      legs.push(chosen);
      cur = chosen.to;
    }
    state.inferencePulse = { legs, currentLeg: 0, progress: 0 };
    src.pulse = 1;
  }
  function updatePulse(dt) {
    const p = state.inferencePulse;
    if (!p) return;
    if (p.legs.length === 0) { state.inferencePulse = null; return; }
    p.progress += dt / 700;
    p.legs[p.currentLeg].activation = Math.min(1, p.legs[p.currentLeg].activation + 0.25);
    if (p.progress >= 1) {
      p.legs[p.currentLeg].to.pulse = 1;
      p.currentLeg++;
      p.progress = 0;
      if (p.currentLeg >= p.legs.length) state.inferencePulse = null;
    }
  }

  /* --- HUD as DOM-text-via-Pixi-Text. Rebuilt every frame. --- */
  const titleText = new PIXI.Text({ text: "Pulse Prune", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 15, fontWeight: "700", fill: COL.text } });
  titleText.anchor.set(0.5, 0); titleText.position.set(W / 2, 14);
  const subtitleText = new PIXI.Text({ text: "click dim weights · keep bright ones · 60% sparsity in 45s", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 10.5, fill: COL.muted } });
  subtitleText.anchor.set(0.5, 0); subtitleText.position.set(W / 2, 36);
  hudLayer.addChild(titleText, subtitleText);

  /* HUD bar containers (drawn imperatively so colour can change with state) */
  const sparsityLabel = new PIXI.Text({ text: "", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 10, fill: 0x555555 } });
  const sparsityBar = new PIXI.Graphics();
  const accuracyLabel = new PIXI.Text({ text: "", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 10, fill: 0x555555 } });
  const accuracyBar = new PIXI.Graphics();
  const timerLabel = new PIXI.Text({ text: "", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 11, fontWeight: "700", fill: 0x555555 } });
  const dailyLabel = new PIXI.Text({ text: "", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 9, fill: 0x999999 } });
  hudLayer.addChild(sparsityLabel, sparsityBar, accuracyLabel, accuracyBar, timerLabel, dailyLabel);

  const gameOverOverlay = new PIXI.Container();
  gameOverOverlay.visible = false;
  overlayLayer.addChild(gameOverOverlay);

  /* --- Frame loop --- */
  let last = performance.now();
  app.ticker.add((ticker) => {
    const dt = ticker.deltaMS;
    if (!state.over) {
      if (state.started) {
        state.timeLeft -= dt;
        if (state.timeLeft <= 0) {
          state.timeLeft = 0;
          state.over = true;
          state.won = state.sparsity >= TARGET_SPARSITY && state.accuracy >= ACCURACY_FLOOR;
          endGame();
        }
      }
      if (!state.inferencePulse) {
        state.pulseCooldown -= dt;
        if (state.pulseCooldown <= 0) {
          spawnPulse();
          state.pulseCooldown = 1200;
        }
      } else {
        updatePulse(dt);
      }
    }
    for (let i = 0; i < weights.length; i++) {
      if (!weights[i].pruned) weights[i].activation *= 0.92;
    }
    for (const n of neurons) n.pulse *= 0.93;

    if (opts.onScoreChange && !state.over) {
      opts.onScoreChange({
        accuracy: state.accuracy,
        sparsity: state.sparsity,
        timeLeft: state.timeLeft,
        started: state.started,
        alltimeBest: alltimeBestRef.v
      });
    }

    redraw();
    if (state.over && !gameOverOverlay.visible) drawGameOver();
  });

  function redraw() {
    /* Edges */
    edgesGfx.clear();
    for (let i = 0; i < weights.length; i++) {
      const w = weights[i];
      if (w.pruned) continue;
      const baseAlpha = Math.max(0.1, Math.min(1, w.magnitude * 0.7));
      const active = w.activation;
      const safeHintPulse = (!state.started && i === safestWeightIdx)
        ? (0.18 + 0.18 * (0.5 + 0.5 * Math.sin(performance.now() * 0.008)))
        : 0;
      const alpha = Math.min(1, baseAlpha + active * 0.5 + safeHintPulse);
      let color;
      if (i === state.hoverIdx) color = COL.mitRed;
      else if (!state.started && i === safestWeightIdx) color = COL.greenStroke;
      else if (w.magnitude > 0.55) color = COL.blueStroke;
      else if (w.magnitude > 0.30) color = COL.blueMid;
      else color = COL.blueFaint;
      const lineWidth = (i === state.hoverIdx)
        ? 3
        : (!state.started && i === safestWeightIdx)
          ? 3.2
          : 1.2 + w.magnitude * 1.8;
      edgesGfx.moveTo(w.from.x, w.from.y);
      edgesGfx.lineTo(w.to.x, w.to.y);
      edgesGfx.stroke({ width: lineWidth, color, alpha });
    }

    /* Inference pulse — head dot (with GlowFilter once filters load) and a
       trailing tail of the last PULSE_TAIL_LEN positions. The tail is drawn
       in pulseTailGfx so it sits behind the head and renders with its own
       (softer) glow. */
    pulseGfx.clear();
    pulseTailGfx.clear();
    if (state.inferencePulse && state.inferencePulse.legs.length > 0) {
      const p = state.inferencePulse;
      const leg = p.legs[p.currentLeg];
      if (leg) {
        const t = Math.max(0, Math.min(1, p.progress));
        const dx = leg.from.x + (leg.to.x - leg.from.x) * t;
        const dy = leg.from.y + (leg.to.y - leg.from.y) * t;
        /* Push current position into the tail buffer */
        pulseTail.push({ x: dx, y: dy });
        while (pulseTail.length > PULSE_TAIL_LEN) pulseTail.shift();
        /* Draw the tail: each older position smaller and fainter */
        for (let i = 0; i < pulseTail.length - 1; i++) {
          const frac = i / (PULSE_TAIL_LEN - 1);
          const pt = pulseTail[i];
          pulseTailGfx.circle(pt.x, pt.y, 1.5 + frac * 3).fill({ color: COL.blueStroke, alpha: frac * 0.45 });
        }
        /* Head dot. The GlowFilter (loaded async) does the heavy lifting on
           bright halo. Fallback radial fakes glow if filters haven't loaded yet. */
        if (filters && filters.GlowFilter) {
          pulseGfx.circle(dx, dy, 5).fill({ color: COL.blueStroke, alpha: 1 });
        } else {
          pulseGfx.circle(dx, dy, 14).fill({ color: COL.blueStroke, alpha: 0.10 });
          pulseGfx.circle(dx, dy, 9).fill({ color: COL.blueStroke, alpha: 0.25 });
          pulseGfx.circle(dx, dy, 5).fill({ color: COL.blueStroke, alpha: 0.55 });
          pulseGfx.circle(dx, dy, 3).fill({ color: COL.blueStroke, alpha: 1 });
        }
      }
    } else {
      /* No active pulse → drain the tail */
      if (pulseTail.length > 0) pulseTail.length = 0;
    }

    /* Neurons */
    neuronsGfx.clear();
    for (const n of neurons) {
      neuronsGfx.circle(n.x, n.y, 9);
      neuronsGfx.fill({ color: COL.blueLight });
      neuronsGfx.stroke({ width: 1.5, color: COL.blueStroke });
      if (n.pulse > 0.05) {
        neuronsGfx.circle(n.x, n.y, 9 + n.pulse * 8);
        neuronsGfx.stroke({ width: 2, color: COL.mitRed, alpha: n.pulse });
      }
    }

    /* HUD bars */
    const barX = 20;
    const barW = W - 40;
    const spY = H - 40, spH = 6;
    const accY = H - 26, accH = 8;

    sparsityBar.clear();
    sparsityBar.roundRect(barX, spY, barW, spH, 3).fill({ color: 0xeeeeee });
    const spFrac = Math.min(1, state.sparsity / TARGET_SPARSITY);
    const spFill = state.sparsity >= TARGET_SPARSITY ? COL.greenStroke : COL.blueMid;
    sparsityBar.roundRect(barX, spY, barW * spFrac, spH, 3).fill({ color: spFill });
    sparsityBar.moveTo(barX + barW, spY - 3).lineTo(barX + barW, spY + spH + 3).stroke({ width: 1.5, color: COL.greenStroke });
    sparsityLabel.text = `trim ${state.sparsity.toFixed(0)}% / ${TARGET_SPARSITY}% goal`;
    sparsityLabel.position.set(barX, spY - 13);

    accuracyBar.clear();
    accuracyBar.roundRect(barX, accY, barW, accH, 4).fill({ color: 0xeeeeee });
    const accFrac = Math.max(0, Math.min(1, state.accuracy / 100));
    let accColor = COL.greenStroke;
    if (state.accuracy < 80) accColor = 0xc87b2a;
    if (state.accuracy < ACCURACY_FLOOR) accColor = COL.red;
    accuracyBar.roundRect(barX, accY, barW * accFrac, accH, 4).fill({ color: accColor });
    const floorX = barX + barW * (ACCURACY_FLOOR / 100);
    accuracyBar.moveTo(floorX, accY - 2).lineTo(floorX, accY + accH + 2).stroke({ width: 1, color: COL.red, alpha: 1 });
    accuracyLabel.text = `accuracy ${state.accuracy.toFixed(1)}% (stay above ${ACCURACY_FLOOR}%)`;
    accuracyLabel.position.set(barX, accY - 13);

    const secs = Math.ceil(state.timeLeft / 1000);
    timerLabel.text = state.started ? `⏱ ${secs}s` : "⏱ starts on first click";
    timerLabel.style.fill = state.started && secs <= 10 ? 0xc44444 : 0x555555;
    timerLabel.anchor.set(1, 0);
    timerLabel.position.set(W - 20, spY - 14);

    dailyLabel.text = state.started
      ? `daily ${today} · day ${dayNumber()} · all-time best ${alltimeBestRef.v}% · R retry`
      : `click a faint weight to begin · all-time best ${alltimeBestRef.v}%`;
    dailyLabel.position.set(barX, accY + accH + 4);
  }

  function drawGameOver() {
    gameOverOverlay.visible = true;
    const bg = new PIXI.Graphics();
    bg.rect(0, 0, W, H).fill({ color: 0xffffff, alpha: 0.92 });
    const titleColor = state.won ? COL.greenStroke : COL.mitRed;
    const titleStr = state.won ? "🏆 network compressed!" : "accuracy collapsed";
    const t1 = new PIXI.Text({ text: titleStr, style: { fontFamily: "Helvetica Neue, Arial", fontSize: 24, fontWeight: "700", fill: titleColor } });
    t1.anchor.set(0.5, 0.5); t1.position.set(W / 2, H / 2 - 20);
    const t2 = new PIXI.Text({ text: `sparsity ${Math.round(state.sparsity)}% · accuracy ${state.accuracy.toFixed(1)}%`, style: { fontFamily: "Helvetica Neue, Arial", fontSize: 14, fill: COL.text } });
    t2.anchor.set(0.5, 0.5); t2.position.set(W / 2, H / 2 + 8);
    const t3 = new PIXI.Text({ text: "tap or press R to retry", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 11, fontStyle: "italic", fill: 0x777777 } });
    t3.anchor.set(0.5, 0.5); t3.position.set(W / 2, H / 2 + 32);
    gameOverOverlay.addChild(bg, t1, t2, t3);
  }

  /* --- Public API --- */
  return {
    id: "prune",
    name: "Pulse Prune",
    ahaLabel: "You just played at",
    ahaText: "Magnitude is a usable proxy for importance (Han et al. 2015). Real pruning fine-tunes after the cut to recover.",
    ahaLink: { href: "https://arxiv.org/abs/1506.02626", label: "Han et al. 2015 →" },
    buildShareText(result) {
      const tag = result.won ? "🏆 compressed" : "✗ diverged";
      return "MLSysBook Playground · Pulse Prune · day " + dayNumber() + "\n" +
        tag + " · " + result.finalSparsity + "% sparsity · " + result.accuracy.toFixed(0) + "% acc\n" +
        result.emojiGrid + "\n" +
        "play → mlsysbook.ai/games/prune/";
    },
    destroy() {
      window.removeEventListener("keydown", handleKeydown);
      canvas.removeEventListener("mousemove", handlePointerMove);
      canvas.removeEventListener("pointerdown", handlePointerDown);
      app.destroy(true, { children: true, texture: true });
    }
  };
}
