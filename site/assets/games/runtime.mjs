/* ============================================================
   MLSysBook Playground — shared Pixi runtime (v2)
   ------------------------------------------------------------
   Every game on Pixi v8 mounts via this module. It owns:
   - Pixi Application bootstrapping onto an existing <canvas>
   - Daily-seed PRNG, day-number, hashString
   - Best-score persistence (localStorage)
   - Aha-card DOM rendering (one-line citation, see design memo)
   - Shared "juice" primitives (pop ring, screen flash) as Pixi
     Containers — every game reuses the same look.
   - Particle + floating-text helpers wired to Pixi
   - Geometry helpers (segment hit-test) ported from old common.js

   The old window.MLSP API (MLSP.bestScore.get/set, MLSP.dayNumber,
   MLSP.showAhaCard, etc) is preserved so legacy canvas games (Roofline,
   Sharp Shot) keep working unchanged during the migration.
   ============================================================ */

import * as PIXI from "/assets/games/vendor/pixi.min.mjs";

/* -----------------------------------------------------------
   Daily seed + best-score (legacy-compatible)
   ----------------------------------------------------------- */
export function hashString(s) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h;
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function dailySeed(gameId) {
  const today = new Date().toISOString().slice(0, 10);
  return { rand: mulberry32(hashString(gameId + "-" + today)), today };
}

export function dayNumber() {
  const launch = new Date("2026-04-22T00:00:00Z");
  const now = new Date();
  return Math.max(1, Math.floor((now - launch) / (1000 * 60 * 60 * 24)) + 1);
}

export const bestScore = {
  get(key) {
    try { return parseInt(localStorage.getItem("mlsp-best-" + key) || "0", 10) || 0; }
    catch (e) { return 0; }
  },
  set(key, val) {
    try { localStorage.setItem("mlsp-best-" + key, String(val)); }
    catch (e) { /* ignore */ }
  }
};

/* -----------------------------------------------------------
   Geometry helpers (used by hit-testing in Prune et al.)
   ----------------------------------------------------------- */
export function distToSegment(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1, dy = y2 - y1;
  const len2 = dx * dx + dy * dy;
  if (len2 === 0) return Math.hypot(px - x1, py - y1);
  let t = ((px - x1) * dx + (py - y1) * dy) / len2;
  if (t < 0) t = 0; else if (t > 1) t = 1;
  return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
}

/* -----------------------------------------------------------
   Aha card — one-line citation, per design memo round-7.
   `text` is the single sentence; `link` is optional reading.
   ----------------------------------------------------------- */
export function showAhaCard(container, label, text, link) {
  if (!container) return;
  if (container.querySelector(".mlsp-aha-card")) return;
  const card = document.createElement("div");
  card.className = "mlsp-aha-card";
  const lbl = document.createElement("span");
  lbl.className = "mlsp-aha-label";
  lbl.textContent = label;
  const p = document.createElement("p");
  p.textContent = text;
  card.appendChild(lbl);
  card.appendChild(p);
  if (link) {
    const a = document.createElement("a");
    a.href = link.href;
    a.textContent = link.label || "read more →";
    a.className = "mlsp-aha-link";
    a.target = "_blank";
    a.rel = "noopener";
    card.appendChild(a);
  }
  container.appendChild(card);
  return card;
}

/* -----------------------------------------------------------
   Pixi mounting — every game starts here. Returns { app, stage,
   width, height, onTick, destroy } so the game can subscribe
   to the ticker and tear down cleanly on retry.
   ----------------------------------------------------------- */
export async function mountPixiOnCanvas(canvas, opts = {}) {
  const W = canvas.width, H = canvas.height;
  const app = new PIXI.Application();
  await app.init({
    canvas: canvas,
    width: W, height: H,
    backgroundColor: opts.bg ?? 0xffffff,
    antialias: true,
    autoDensity: true,
    resolution: Math.min(2, window.devicePixelRatio || 1)
  });
  // Pixi v8 sets stage.eventMode globally on stage; we want canvas to receive pointer events.
  app.stage.eventMode = "static";
  app.stage.hitArea = app.screen;
  if (opts.backdrop !== false) {
    drawBrandBackdrop(app.stage, W, H, opts);
  }

  const tickHandlers = [];
  app.ticker.add((ticker) => {
    const dt = ticker.deltaMS;
    for (let i = 0; i < tickHandlers.length; i++) tickHandlers[i](dt);
  });

  return {
    app,
    stage: app.stage,
    width: W,
    height: H,
    PIXI,
    onTick: (fn) => tickHandlers.push(fn),
    destroy: () => app.destroy(true, { children: true, texture: true }),
  };
}

export function drawBrandBackdrop(stage, W, H, opts = {}) {
  const bg = opts.bg ?? 0xffffff;
  const r = (bg >> 16) & 255;
  const g = (bg >> 8) & 255;
  const b = bg & 255;
  const isDark = (0.2126 * r + 0.7152 * g + 0.0722 * b) < 90;
  const layer = new PIXI.Graphics();

  if (isDark) {
    layer.rect(0, 0, W, H).fill({ color: bg });
    layer.rect(0, 0, 5, H).fill({ color: 0xa31f34, alpha: 0.85 });
    for (let x = 48; x < W; x += 56) {
      layer.moveTo(x, 32).lineTo(x, H - 32).stroke({ color: 0xffffff, alpha: 0.045, width: 1 });
    }
    for (let y = 42; y < H; y += 46) {
      layer.moveTo(28, y).lineTo(W - 28, y).stroke({ color: 0xffffff, alpha: 0.05, width: 1 });
    }
  } else {
    layer.rect(0, 0, W, H).fill({ color: bg });
    layer.rect(0, 0, 5, H).fill({ color: 0xa31f34, alpha: 0.82 });
    for (let x = 48; x < W; x += 56) {
      layer.moveTo(x, 32).lineTo(x, H - 32).stroke({ color: 0xe9edf2, width: 1 });
    }
    for (let y = 42; y < H; y += 46) {
      layer.moveTo(28, y).lineTo(W - 28, y).stroke({ color: 0xeef1f4, width: 1 });
    }
  }

  stage.addChild(layer);
  return layer;
}

/* -----------------------------------------------------------
   Juice — shared visual feedback using Pixi.
   pop(stage, x, y, color, opts?) draws a quick ring;
   flash(stage, color, ms) overlays a fading rectangle.
   These return objects you can ignore; they self-destruct.
   ----------------------------------------------------------- */
export function pop(stage, x, y, color, opts = {}) {
  const ring = new PIXI.Graphics();
  const r0 = opts.r ?? 16;
  const ms = opts.ms ?? 360;
  const startTime = performance.now();
  ring.position.set(x, y);
  stage.addChild(ring);
  function tick() {
    const t = (performance.now() - startTime) / ms;
    if (t >= 1) { ring.destroy(); return false; }
    const eased = 1 + (2.7) * Math.pow(t - 1, 3) + 1.7 * Math.pow(t - 1, 2);
    const radius = r0 * Math.max(0, eased);
    ring.clear();
    ring.circle(0, 0, radius);
    ring.stroke({ width: 2.2 * (1 - t), color: color, alpha: 1 - (1 - Math.pow(1 - t, 3)) });
    return true;
  }
  // Self-tick via app's shared ticker — caller doesn't need to manage it
  const handler = () => { if (!tick()) PIXI.Ticker.shared.remove(handler); };
  PIXI.Ticker.shared.add(handler);
  return ring;
}

export function flash(stage, color, ms = 220, alpha = 0.40) {
  // Read screen size from stage's first child... safer: caller passes via mount object.
  // Here we use a fullscreen rect via stage hitArea bounds.
  const bounds = stage.hitArea?.getBounds?.() || { x: 0, y: 0, width: 680, height: 460 };
  const overlay = new PIXI.Graphics();
  overlay.rect(bounds.x, bounds.y, bounds.width, bounds.height);
  overlay.fill({ color: color, alpha: alpha });
  stage.addChild(overlay);
  const startTime = performance.now();
  const handler = () => {
    const t = (performance.now() - startTime) / ms;
    if (t >= 1) {
      overlay.destroy();
      PIXI.Ticker.shared.remove(handler);
      return;
    }
    overlay.alpha = (1 - t) * alpha;
  };
  PIXI.Ticker.shared.add(handler);
  return overlay;
}

/* -----------------------------------------------------------
   Particle burst — small ephemeral sprites that fan out and fade.
   ----------------------------------------------------------- */
export function burst(stage, x, y, color, count = 8, spreadOpts = {}) {
  const speed = spreadOpts.speed ?? 2.0;
  const lifeMs = spreadOpts.lifeMs ?? 600;
  for (let i = 0; i < count; i++) {
    const ang = Math.random() * Math.PI * 2;
    const spd = 1 + Math.random() * speed;
    const p = new PIXI.Graphics();
    p.circle(0, 0, 2);
    p.fill({ color });
    p.position.set(x, y);
    let vx = Math.cos(ang) * spd;
    let vy = Math.sin(ang) * spd - 0.4;
    let age = 0;
    stage.addChild(p);
    const handler = (ticker) => {
      const dt = ticker.deltaMS;
      age += dt;
      if (age >= lifeMs) { p.destroy(); PIXI.Ticker.shared.remove(handler); return; }
      vy += 0.15;
      p.position.x += vx;
      p.position.y += vy;
      p.alpha = 1 - age / lifeMs;
    };
    PIXI.Ticker.shared.add(handler);
  }
}

/* -----------------------------------------------------------
   Floating text — short message that drifts up and fades out.
   ----------------------------------------------------------- */
export function floatText(stage, x, y, text, color, opts = {}) {
  const lifeMs = opts.lifeMs ?? 1100;
  const t = new PIXI.Text({
    text,
    style: {
      fontFamily: "Helvetica Neue, Arial, sans-serif",
      fontSize: opts.size ?? 12,
      fontWeight: "700",
      fill: color,
      align: "center"
    }
  });
  t.anchor.set(0.5, 0.5);
  t.position.set(x, y);
  stage.addChild(t);
  let age = 0;
  const handler = (ticker) => {
    const dt = ticker.deltaMS;
    if (!t || t.destroyed || !t.position) { PIXI.Ticker.shared.remove(handler); return; }
    age += dt;
    if (age >= lifeMs) { t.destroy(); PIXI.Ticker.shared.remove(handler); return; }
    t.position.y -= dt * 0.035;
    t.alpha = 1 - age / lifeMs;
  };
  PIXI.Ticker.shared.add(handler);
}

/* -----------------------------------------------------------
   Tween helper — tween a numeric property of a Pixi DisplayObject
   from a start to end value with easing. Returns a function you
   can call to cancel.
   ----------------------------------------------------------- */
const easings = {
  linear: t => t,
  outCubic: t => 1 - Math.pow(1 - t, 3),
  outBack: t => { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); },
  outElastic: t => {
    const c4 = (2 * Math.PI) / 3;
    return t === 0 ? 0 : t === 1 ? 1 : Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
  },
  outExpo: t => t === 1 ? 1 : 1 - Math.pow(2, -10 * t),
};

export function tween(target, prop, from, to, ms, ease = "outCubic") {
  if (Array.isArray(prop)) {
    // Tween multiple props in lockstep
    const startTime = performance.now();
    const fn = easings[ease] || easings.outCubic;
    const handler = (ticker) => {
      const t = Math.max(0, Math.min(1, (performance.now() - startTime) / ms));
      const e = fn(t);
      for (let i = 0; i < prop.length; i++) {
        const path = prop[i].split(".");
        let obj = target;
        for (let j = 0; j < path.length - 1; j++) obj = obj[path[j]];
        obj[path[path.length - 1]] = from[i] + (to[i] - from[i]) * e;
      }
      if (t >= 1) PIXI.Ticker.shared.remove(handler);
    };
    PIXI.Ticker.shared.add(handler);
    return () => PIXI.Ticker.shared.remove(handler);
  } else {
    const startTime = performance.now();
    const fn = easings[ease] || easings.outCubic;
    const path = prop.split(".");
    const handler = (ticker) => {
      const t = Math.max(0, Math.min(1, (performance.now() - startTime) / ms));
      const e = fn(t);
      let obj = target;
      for (let j = 0; j < path.length - 1; j++) obj = obj[path[j]];
      obj[path[path.length - 1]] = from + (to - from) * e;
      if (t >= 1) PIXI.Ticker.shared.remove(handler);
    };
    PIXI.Ticker.shared.add(handler);
    return () => PIXI.Ticker.shared.remove(handler);
  }
}

/* -----------------------------------------------------------
   Lazy-loaded filters — pixi-filters bundle is ~210kB so we
   only fetch it when a game actually asks for filters. Returns
   a Promise that resolves to the filter namespace.
   ----------------------------------------------------------- */
let _filtersPromise = null;
export function getFilters() {
  if (!_filtersPromise) {
    _filtersPromise = import("/assets/games/vendor/pixi-filters.min.mjs");
  }
  return _filtersPromise;
}

/* -----------------------------------------------------------
   Camera-shake helper — applies a temporary jitter offset
   to a target Pixi container (typically the game-layer container,
   not the whole stage, so HUD stays still).
   ----------------------------------------------------------- */
export function shake(target, amount, ms) {
  if (!target.__mlspShake) target.__mlspShake = { amt: 0, t: 0, basex: target.position.x, basey: target.position.y };
  target.__mlspShake.basex = target.position.x - (target.__mlspShake.shakeX || 0);
  target.__mlspShake.basey = target.position.y - (target.__mlspShake.shakeY || 0);
  target.__mlspShake.amt = Math.max(target.__mlspShake.amt, amount);
  target.__mlspShake.t = Math.max(target.__mlspShake.t, ms);
  if (target.__mlspShake.handler) return;
  target.__mlspShake.handler = (ticker) => {
    const dt = ticker.deltaMS;
    const s = target.__mlspShake;
    s.t -= dt;
    if (s.t <= 0) {
      target.position.set(s.basex, s.basey);
      s.amt = 0;
      s.shakeX = 0; s.shakeY = 0;
      PIXI.Ticker.shared.remove(s.handler);
      s.handler = null;
      return;
    }
    s.shakeX = (Math.random() - 0.5) * s.amt;
    s.shakeY = (Math.random() - 0.5) * s.amt;
    target.position.set(s.basex + s.shakeX, s.basey + s.shakeY);
  };
  PIXI.Ticker.shared.add(target.__mlspShake.handler);
}

/* -----------------------------------------------------------
   Legacy bridge — expose runtime on window.MLSP so the
   old common.js callers (Roofline, Sharp Shot) and the qmd
   boot scripts keep working without modification.
   ----------------------------------------------------------- */
window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};
window.MLSP.bestScore = bestScore;
window.MLSP.dayNumber = dayNumber;
window.MLSP.hashString = hashString;
window.MLSP.mulberry32 = mulberry32;
window.MLSP.distToSegment = distToSegment;
window.MLSP.showAhaCard = showAhaCard;
window.MLSP.PIXI = PIXI;
window.MLSP.runtime = {
  mountPixiOnCanvas,
  pop, flash, burst, floatText, shake, tween, getFilters,
  dailySeed
};

if (!window.MLSP.__fullscreenToggleBound) {
  window.MLSP.__fullscreenToggleBound = true;
  document.addEventListener("click", (event) => {
    const button = event.target.closest?.(".mlsp-fullscreen-btn");
    if (!button) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    const container = button.closest(".mlsp-game-container");
    if (!container) return;
    if (document.fullscreenElement) {
      document.exitFullscreen?.();
    } else {
      container.requestFullscreen?.();
    }
  }, true);
}

/* -----------------------------------------------------------
   Page lifecycle: trigger any registered onReady() callback
   so games waiting for runtime can boot themselves.
   ----------------------------------------------------------- */
window.MLSP.runtimeReady = true;
const evt = new CustomEvent("mlsp:runtime-ready");
window.dispatchEvent(evt);
