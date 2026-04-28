// Module-relative imports so the game works at any deploy base
// (mlsysbook.ai/, harvard-edge.github.io/cs249r_book_dev/, localhost:N/, …).
// Absolute "/assets/..." paths break on any non-root deployment.
import { mountPixiOnCanvas, burst, floatText, flash, shake, dailySeed } from "./runtime.mjs";
import { bestScore, dayNumber } from "./runtime.mjs";
import * as P from "./vendor/pixi.min.mjs";

window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};
window.MLSP.games.lander = function(canvas, opts) { return mountLander(canvas, opts); };

export async function mountLander(canvas, opts = {}) {
  const { app, stage, width: W, height: H } = await mountPixiOnCanvas(canvas, { bg: 0xfbfbfb });

  const COL = {
    bg: 0xfbfbfb, text: 0x333333,
    ship: 0x4a90c4, thrust: 0xc87b2a,
    pad: 0x3d9e5a, localPad: 0xc87b2a, ground: 0x6f7782,
    crash: 0xc44444
  };

  const state = {
    x: W / 2, y: 40,
    vx: 0, vy: 0,
    angle: 0, // radians
    fuel: 100, // VRAM
    over: false, won: false,
    started: false,
    reason: null,           // 'win' | 'diverged' | 'local-min' | 'off-course' | 'missed-basin' | 'oom'
    gameOverFired: false,   // guard so onGameOver fires exactly once
    keys: { up: false, left: false, right: false }
  };

  // Per-failure aha messages — every loss type maps to a distinct ML systems lesson.
  // The qmd's attachAha pulls these via api.aha(reason).
  const AHA = {
    win: {
      label: "You just experienced",
      text: "A balanced descent. The right learning rate steered you into the deep basin while a measured batch size kept VRAM in budget. In real training, that's the dream — fast convergence without the OOM tax."
    },
    diverged: {
      label: "What just happened",
      text: "You hit the global minimum at high speed. In SGD this is what an over-aggressive learning rate looks like: the optimizer overshoots the basin and the loss diverges. Smaller learning rate → softer touchdown."
    },
    "local-min": {
      label: "What just happened",
      text: "You settled in a basin — but not the deep one. Real loss landscapes have many local minima; without enough exploration (gradient noise, momentum, restarts), an optimizer can stop short of the optimum."
    },
    "off-course": {
      label: "What just happened",
      text: "Your update steps drifted out of the parameter space the model can handle. In practice this is what divergent training looks like: weights blow up, gradients overflow, and the run is unrecoverable."
    },
    "missed-basin": {
      label: "What just happened",
      text: "You touched down in a region with no basin — a flat or saddle area of the loss surface. The optimizer has no clear gradient signal to follow, and the model never learns the task well."
    },
    oom: {
      label: "What just happened",
      text: "You burned through your VRAM mid-run. Every increase in batch size compounds the working memory; once you OOM, the process is dead. In production, this is the brutal practical ceiling on large-batch training."
    }
  };

  const gravity = 0.05;
  const thrustPower = 0.15;
  const rotSpeed = 0.055;            // tuned down from 0.08 — easier precision, still feels responsive
  const maxSafeSpeed = 2.4;          // tuned up from 2.0 — first-time forgiveness, still teaches the lesson
  const maxSafeAngle = 0.6;          // tuned up from 0.5 (~34°) — same reasoning

  // Daily seed: every player worldwide gets the same loss landscape today.
  // Reset at UTC midnight via dailySeed() implementation.
  const seed = dailySeed("lander");
  const day = dayNumber();
  const seededRand = seed.rand;
  const terrain = createTerrain();

  // Honor the user's OS-level reduced-motion preference. Disables four animations
  // (goal-pad pulse, CTA pulse, camera shake, particle bursts on losses).
  const reduceMotion =
    typeof window !== "undefined" &&
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const safeShake = (target, amount, ms) => { if (!reduceMotion) shake(target, amount, ms); };
  const safeBurst = (s, x, y, c, n, o) => { if (!reduceMotion) burst(s, x, y, c, n, o); };

  function rand(min, max) {
    return min + seededRand() * (max - min);
  }

  function createTerrain() {
    // Implicit level system: difficulty scales gently with day number.
    // Day 1 → level 0 (mild). Day 11+ → level 8 (max). Caps so it never gets impossible.
    const level = Math.min(8, Math.max(0, day - 1));

    // Global pad now spawns anywhere across the playable width (was always near center).
    // Forces the player to actually steer instead of drop-and-tap.
    const globalX = rand(W * 0.18, W * 0.82);

    // Local pads always flank global with a minimum gap so wells don't bleed together.
    // If only one half has room, both locals go there, separated within that half.
    const minGap = 110;        // center-to-center gap from global pad
    const edge   = 60;         // canvas-edge margin for any pad
    const leftHalf  = [edge, globalX - minGap];
    const rightHalf = [globalX + minGap, W - edge];
    const leftHasRoom  = leftHalf[1]  - leftHalf[0]  > 60;
    const rightHasRoom = rightHalf[1] - rightHalf[0] > 60;
    let leftX, rightX;
    if (leftHasRoom && rightHasRoom) {
      leftX  = rand(leftHalf[0],  leftHalf[1]);
      rightX = rand(rightHalf[0], rightHalf[1]);
    } else if (rightHasRoom) {
      const w = rightHalf[1] - rightHalf[0];
      leftX  = rand(rightHalf[0], rightHalf[0] + w * 0.4);
      rightX = rand(rightHalf[1] - w * 0.4, rightHalf[1]);
    } else {
      const w = leftHalf[1] - leftHalf[0];
      leftX  = rand(leftHalf[0], leftHalf[0] + w * 0.4);
      rightX = rand(leftHalf[1] - w * 0.4, leftHalf[1]);
    }

    return {
      phase: rand(0, Math.PI * 2),
      slope: rand(-4, 4),  // tuned down from rand(-8, 8) — caps random unfairness from extreme tilts
      level,
      global: { x: globalX, width: rand(86, 112), wellWidth: rand(72, 92), depth: rand(50, 66) },
      locals: [
        { x: leftX,  width: rand(70, 92), wellWidth: rand(54, 70), depth: rand(24, 36) },
        { x: rightX, width: rand(70, 92), wellWidth: rand(54, 70), depth: rand(24, 36) }
      ]
    };
  }

  function well(x, center, width, depth) {
    const z = (x - center) / width;
    return depth * Math.exp(-z * z);
  }

  function lossY(x) {
    const normalized = x / W;
    // Difficulty-scaled curvature: amplitudes grow with level so higher levels feel
    // more rugged. Third harmonic adds finer ripples that weren't in the original.
    const a1 = 13 + terrain.level * 1.6;
    const a2 = 7  + terrain.level * 1.2;
    const a3 = 4  + terrain.level * 0.9;   // new third harmonic — adds finer surface texture
    const base = H - 96
      + terrain.slope * (normalized - 0.5)
      + a1 * Math.sin(normalized * Math.PI * 2.1 + terrain.phase)
      + a2 * Math.sin(normalized * Math.PI * 4.7 + terrain.phase * 0.53)
      + a3 * Math.sin(normalized * Math.PI * 7.3 + terrain.phase * 1.31);
    return base
      + well(x, terrain.global.x, terrain.global.wellWidth, terrain.global.depth)
      + well(x, terrain.locals[0].x, terrain.locals[0].wellWidth, terrain.locals[0].depth)
      + well(x, terrain.locals[1].x, terrain.locals[1].wellWidth, terrain.locals[1].depth);
  }

  function inPad(x, pad) {
    return x >= pad.x - pad.width / 2 && x <= pad.x + pad.width / 2;
  }
  function inGlobalPad(x) { return inPad(x, terrain.global); }
  function inLocalPad(x) {
    return terrain.locals.some(pad => inPad(x, pad));
  }

  const bg = new P.Graphics();
  bg.rect(0, 0, W, H).fill({ color: COL.bg });
  for (let y = 70; y < H - 70; y += 42) {
    bg.moveTo(26, y).lineTo(W - 26, y).stroke({ color: 0xe7eaee, width: 1 });
  }
  for (let x = 40; x < W; x += 70) {
    bg.moveTo(x, 54).lineTo(x, H - 54).stroke({ color: 0xf0f2f4, width: 1 });
  }

  const basinWash = new P.Graphics();
  basinWash.moveTo(0, lossY(0));
  for (let x = 0; x <= W; x += 10) basinWash.lineTo(x, lossY(x));
  basinWash.lineTo(W, H).lineTo(0, H).closePath().fill({ color: 0xeaf2f8 });

  const contourLayer = new P.Graphics();
  for (let offset = 18; offset <= 72; offset += 18) {
    contourLayer.moveTo(0, lossY(0) - offset);
    for (let x = 0; x <= W; x += 10) contourLayer.lineTo(x, lossY(x) - offset);
    contourLayer.stroke({ color: offset === 18 ? 0xd9e7f1 : 0xedf2f5, width: 1 });
  }

  const surface = new P.Graphics();
  surface.moveTo(0, lossY(0));
  for (let x = 0; x <= W; x += 6) surface.lineTo(x, lossY(x));
  surface.stroke({ color: COL.ground, width: 3 });

  const padGlobal = new P.Graphics();
  padGlobal.roundRect(terrain.global.x - terrain.global.width / 2, lossY(terrain.global.x) - 3, terrain.global.width, 6, 3).fill({ color: COL.pad });
  const padLocal1 = new P.Graphics();
  padLocal1.roundRect(terrain.locals[0].x - terrain.locals[0].width / 2, lossY(terrain.locals[0].x) - 3, terrain.locals[0].width, 6, 3).fill({ color: COL.localPad });
  const padLocal2 = new P.Graphics();
  padLocal2.roundRect(terrain.locals[1].x - terrain.locals[1].width / 2, lossY(terrain.locals[1].x) - 3, terrain.locals[1].width, 6, 3).fill({ color: COL.localPad });

  const globalLabel = new P.Text({ text: "GLOBAL MINIMUM", style: { fill: 0x3d9e5a, fontSize: 11, fontWeight: "800", letterSpacing: 1 }});
  globalLabel.anchor.set(0.5);
  globalLabel.position.set(terrain.global.x, lossY(terrain.global.x) - 20);
  // Both local pads now labeled symmetrically (right one was previously mute).
  const localLabel1 = new P.Text({ text: "local min", style: { fill: 0x9a6620, fontSize: 10, fontWeight: "600" }});
  localLabel1.anchor.set(0.5);
  localLabel1.position.set(terrain.locals[0].x, lossY(terrain.locals[0].x) - 17);
  const localLabel2 = new P.Text({ text: "local min", style: { fill: 0x9a6620, fontSize: 10, fontWeight: "600" }});
  localLabel2.anchor.set(0.5);
  localLabel2.position.set(terrain.locals[1].x, lossY(terrain.locals[1].x) - 17);

  stage.addChild(bg, contourLayer, basinWash, surface, padGlobal, padLocal1, padLocal2, globalLabel, localLabel1, localLabel2);

  // Predictive trajectory marker — translucent target showing where the ship will be in
  // ~0.5 s if the player keeps coasting (no thrust). Helps a new player anticipate the basin.
  const traj = new P.Graphics();
  stage.addChild(traj);

  // Altitude reference: faint vertical line from ship to the ground directly below.
  // Reads as a "how high am I" cue without competing with the trajectory arrow.
  const altLine = new P.Graphics();
  stage.addChild(altLine);

  // Ship — layered silhouette so the protagonist of the screen reads cleanly.
  // Soft outer halo + crisp body + interior highlight stripe.
  const ship = new P.Graphics();
  // outer halo
  ship.moveTo(0, -17).lineTo(12, 12).lineTo(-12, 12).closePath().fill({ color: COL.ship, alpha: 0.18 });
  // body
  ship.moveTo(0, -15).lineTo(10, 10).lineTo(-10, 10).closePath().fill({ color: COL.ship });
  // crisp outline
  ship.moveTo(0, -15).lineTo(10, 10).lineTo(-10, 10).closePath().stroke({ color: 0x2c5775, width: 1.5 });
  // inner highlight (catches the eye even at small scale)
  ship.moveTo(0, -11).lineTo(4, 6).lineTo(-4, 6).closePath().fill({ color: 0xeaf3fa, alpha: 0.65 });
  stage.addChild(ship);

  // Flame — layered for depth: outer glow + core, both visible only when thrusting.
  const flame = new P.Graphics();
  // outer glow
  flame.moveTo(-7, 10).lineTo(0, 30).lineTo(7, 10).closePath().fill({ color: COL.thrust, alpha: 0.35 });
  // core
  flame.moveTo(-4, 10).lineTo(0, 23).lineTo(4, 10).closePath().fill({ color: 0xffd28a });
  flame.visible = false;
  ship.addChild(flame);

  // Loss-landscape legend — anchored to the basin so it reads as chart annotation,
  // not a free-floating label. Less visual noise, more pedagogical signal.
  const noiseText = new P.Text({
    text: "loss landscape",
    style: { fill: 0x6f7782, fontSize: 11, fontWeight: "700", letterSpacing: 1 }
  });
  noiseText.position.set(18, H - 100);
  stage.addChild(noiseText);

  // Daily-puzzle + personal-best chip (top-center). Reads as "you're on day 4,
  // your softest landing today was 1.4 — try to beat it."
  const dayChip = new P.Text({
    text: bestSoftText(),
    style: { fill: 0x6f7782, fontSize: 11, fontWeight: "600", letterSpacing: 0.5, align: "center" }
  });
  dayChip.anchor.set(0.5, 0);
  dayChip.position.set(W / 2, 16);
  stage.addChild(dayChip);

  function bestSoftText() {
    const bestRaw = bestScore.get("lander-soft");
    const lvl = `LVL ${1 + terrain.level}`;
    if (!bestRaw) return `Day #${day} · ${lvl} · land softer than yesterday`;
    // Stored as impact-speed × 100 (screen-velocity, dimensionless). Show as v=X.XX.
    const v = (bestRaw / 100).toFixed(2);
    return `Day #${day} · ${lvl} · your softest landing: v=${v}`;
  }

  // ── In-canvas HUD ── glanceable bars so the player never has to look away from the ship.
  // VRAM (vertical, top-right). Color shifts blue → orange → red as memory depletes.
  const vramBar = new P.Container();
  vramBar.position.set(W - 36, 60);
  const vramBarBg = new P.Graphics();
  vramBarBg.rect(0, 0, 14, 200).fill({ color: 0xeef1f4 }).stroke({ color: 0xcfd5db, width: 1 });
  vramBar.addChild(vramBarBg);
  const vramBarFill = new P.Graphics();
  vramBar.addChild(vramBarFill);
  const vramLabel = new P.Text({ text: "VRAM", style: { fill: 0x666666, fontSize: 10, fontWeight: "700", letterSpacing: 1 }});
  vramLabel.anchor.set(0.5, 1);
  vramLabel.position.set(7, -4);
  vramBar.addChild(vramLabel);
  stage.addChild(vramBar);

  // Speed (horizontal, bottom-left). Green safe-band, red danger-band, ticking marker.
  const speedBar = new P.Container();
  speedBar.position.set(22, H - 36);
  const speedTrack = new P.Graphics();
  // Safe zone (0..maxSafeSpeed), then danger zone — drawn once and held.
  const speedW = 220;
  const safeFrac = maxSafeSpeed / 4.5;          // 4.5 = visible-speed cap
  speedTrack.rect(0, 0, speedW * safeFrac, 8).fill({ color: 0xd4edda }).stroke({ color: 0x3d9e5a, width: 1 });
  speedTrack.rect(speedW * safeFrac, 0, speedW * (1 - safeFrac), 8).fill({ color: 0xf9d6d5 }).stroke({ color: 0xc44, width: 1 });
  speedBar.addChild(speedTrack);
  const speedTick = new P.Graphics();
  speedBar.addChild(speedTick);
  const speedLabel = new P.Text({ text: "DESCENT SPEED", style: { fill: 0x666666, fontSize: 10, fontWeight: "700", letterSpacing: 1 }});
  speedLabel.position.set(0, -16);
  speedBar.addChild(speedLabel);
  const safeMarker = new P.Text({ text: "soft-landing limit ↑", style: { fill: 0x3d9e5a, fontSize: 9, fontWeight: "600" }});
  safeMarker.anchor.set(0.5, 0);
  safeMarker.position.set(speedW * safeFrac, 12);
  speedBar.addChild(safeMarker);
  stage.addChild(speedBar);

  function drawHud() {
    const fuelFrac = Math.max(0, Math.min(1, state.fuel / 100));
    const filledH = 200 * fuelFrac;
    const fillColor = fuelFrac > 0.5 ? 0x4a90c4 : (fuelFrac > 0.2 ? 0xc87b2a : 0xc44444);
    vramBarFill.clear();
    vramBarFill.rect(1, 1 + (200 - filledH), 12, filledH).fill({ color: fillColor });

    const speed = Math.hypot(state.vx, state.vy);
    const tickX = Math.min(speedW, (speed / 4.5) * speedW);
    speedTick.clear();
    speedTick.rect(tickX - 1, -3, 3, 14).fill({ color: 0x101827 });
  }

  // ── Goal-pad pulse: the green platform breathes so the player's eye lands on it first ──
  const padPulse = new P.Graphics();
  padPulse.position.set(terrain.global.x, lossY(terrain.global.x));
  stage.addChildAt(padPulse, stage.getChildIndex(padGlobal));
  let pulseT = 0;

  // ── RETRY pill: appears after game-over, clickable + responds to R key ──
  const retryBtn = new P.Container();
  retryBtn.position.set(W / 2, H / 2 + 50);
  retryBtn.eventMode = "static";
  retryBtn.cursor = "pointer";
  retryBtn.visible = false;
  const retryBg = new P.Graphics();
  retryBg.roundRect(-78, -18, 156, 36, 18).fill({ color: 0xa31f34 }).stroke({ color: 0x6f1424, width: 1.5 });
  retryBtn.addChild(retryBg);
  const retryLbl = new P.Text({ text: "↺  TRY AGAIN", style: { fill: 0xffffff, fontSize: 14, fontWeight: "700", letterSpacing: 1 }});
  retryLbl.anchor.set(0.5);
  retryBtn.addChild(retryLbl);
  retryBtn.on("pointertap", () => { if (opts.onRetry) opts.onRetry(); });
  // Added to stage AFTER the READY overlay block so its z-order is on top.

  // ── READY overlay: full-canvas card the player must dismiss with ↑ ──
  const ready = new P.Container();
  const readyDim = new P.Graphics();
  readyDim.rect(0, 0, W, H).fill({ color: 0x101827, alpha: 0.78 });
  ready.addChild(readyDim);
  const readyTitle = new P.Text({ text: "GRADIENT LANDER", style: { fill: 0xffffff, fontSize: 32, fontWeight: "800", letterSpacing: 2 }});
  readyTitle.anchor.set(0.5);
  readyTitle.position.set(W / 2, H / 2 - 70);
  ready.addChild(readyTitle);
  const readyGoal = new P.Text({ text: "Land softly on the GREEN pad — the global minimum.", style: { fill: 0xd4edda, fontSize: 16 }});
  readyGoal.anchor.set(0.5);
  readyGoal.position.set(W / 2, H / 2 - 30);
  ready.addChild(readyGoal);
  const readyControls = new P.Text({
    text: "↑  hold to thrust  (burns VRAM)\n← →  steer learning rate",
    style: { fill: 0xffffff, fontSize: 15, align: "center", lineHeight: 22 }
  });
  readyControls.anchor.set(0.5);
  readyControls.position.set(W / 2, H / 2 + 14);
  ready.addChild(readyControls);
  const readyHint = new P.Text({
    text: "Take your time — read the controls.",
    style: { fill: 0xb8c2cc, fontSize: 12, fontStyle: "italic" }
  });
  readyHint.anchor.set(0.5);
  readyHint.position.set(W / 2, H / 2 + 56);
  ready.addChild(readyHint);
  const readyCta = new P.Text({
    text: "press  ENTER  to launch",
    style: { fill: 0xffd6a8, fontSize: 18, fontWeight: "700", letterSpacing: 1.5 }
  });
  readyCta.anchor.set(0.5);
  readyCta.position.set(W / 2, H / 2 + 86);
  ready.addChild(readyCta);
  stage.addChild(ready);
  stage.addChild(retryBtn);    // retry pill goes on top of everything else
  let ctaPulseT = 0;

  function launch() {
    if (!state.started) { state.started = true; ready.visible = false; }
  }

  const handleKeydown = (e) => {
    // Pre-game launch: accept Enter (primary), Space, or ↑ — any of the three works
    // so the player isn't blocked by guessing the "right" key.
    if (!state.started && (e.key === 'Enter' || e.key === ' ' || e.key === 'ArrowUp')) {
      e.preventDefault();
      launch();
      // ↑ should ALSO start thrusting (preserves existing keyboard pattern); Enter/Space
      // just launch and wait for the player to press ↑.
      if (e.key === 'ArrowUp') state.keys.up = true;
      return;
    }
    if (e.key === 'ArrowUp')    { e.preventDefault(); state.keys.up = true; }
    if (e.key === 'ArrowLeft')  { e.preventDefault(); state.keys.left = true; }
    if (e.key === 'ArrowRight') { e.preventDefault(); state.keys.right = true; }
    if (e.key.toLowerCase() === 'r' && state.over && opts.onRetry) opts.onRetry();
  };
  const handleKeyup = (e) => {
    if (e.key === 'ArrowUp') state.keys.up = false;
    if (e.key === 'ArrowLeft') state.keys.left = false;
    if (e.key === 'ArrowRight') state.keys.right = false;
  };
  window.addEventListener('keydown', handleKeydown);
  window.addEventListener('keyup', handleKeyup);

  // ── Touch zones (mobile / pointer): three invisible Pixi rects mapping to ←, →, ↑.
  // Center zone doubles as the "press to launch" trigger so the READY card responds
  // to a tap as well as the keyboard ↑.
  function makeZone(x, y, w, h, onDown, onUp) {
    const z = new P.Graphics();
    z.rect(x, y, w, h).fill({ color: 0x000000, alpha: 0 });
    z.eventMode = "static";
    z.cursor = "default";
    z.on("pointerdown",  onDown);
    z.on("pointerup",    onUp);
    z.on("pointerupoutside", onUp);
    z.on("pointercancel",     onUp);
    stage.addChild(z);
    return z;
  }
  const colW = W / 3;
  makeZone(0, 0, colW, H,
    () => { state.keys.left = true; },
    () => { state.keys.left = false; });
  makeZone(W - colW, 0, colW, H,
    () => { state.keys.right = true; },
    () => { state.keys.right = false; });
  makeZone(colW, 0, colW, H,
    () => { state.keys.up = true; launch(); },
    () => { state.keys.up = false; });

  // Re-pin overlay UI to the top so touch zones don't intercept clicks meant
  // for the READY overlay or the RETRY button.
  stage.setChildIndex(ready, stage.children.length - 1);
  stage.setChildIndex(retryBtn, stage.children.length - 1);

  app.ticker.add(() => {
    // HUD drawn every frame so it never goes stale between win/crash and the aha card.
    drawHud();

    // Goal pad pulses to attract the eye — held static when reduce-motion is on
    // (a static green ring still reads as the goal; the breathing is decoration).
    if (!reduceMotion) {
      pulseT += 0.06;
      const r = 26 + Math.sin(pulseT) * 6;
      padPulse.clear();
      padPulse.circle(0, 0, r).stroke({ color: COL.pad, width: 2, alpha: 0.55 });
      padPulse.circle(0, 0, r * 0.55).stroke({ color: COL.pad, width: 1.4, alpha: 0.35 });
    } else if (padPulse.geometry == null || pulseT === 0) {
      padPulse.clear();
      padPulse.circle(0, 0, 28).stroke({ color: COL.pad, width: 2, alpha: 0.5 });
      pulseT = -1; // sentinel: drawn once, never again
    }

    if (!state.started) {
      // Pre-game: pulse the "press UP" prompt so it reads as an action, not a static label.
      if (!reduceMotion) {
        ctaPulseT += 0.08;
        readyCta.alpha = 0.75 + 0.25 * Math.sin(ctaPulseT);
      } else {
        readyCta.alpha = 1.0;
      }
      return;
    }
    if (state.over) return;

    if (state.keys.left) state.angle -= rotSpeed;
    if (state.keys.right) state.angle += rotSpeed;

    state.vy += gravity;

    if (state.keys.up && state.fuel > 0) {
      state.vx += Math.sin(state.angle) * thrustPower;
      state.vy -= Math.cos(state.angle) * thrustPower;
      state.fuel -= 0.2;
      flame.visible = true;
    } else {
      flame.visible = false;
    }

    state.x += state.vx;
    state.y += state.vy;

    ship.x = state.x;
    ship.y = state.y;
    ship.rotation = state.angle;

    // Trajectory marker: project 30 frames ahead under coast (gravity only, no thrust).
    const tAhead = 30;
    const predX = state.x + state.vx * tAhead;
    const predY = state.y + state.vy * tAhead + 0.5 * gravity * tAhead * tAhead;
    traj.clear();
    traj.circle(predX, predY, 5).stroke({ color: COL.ship, width: 1.5, alpha: 0.55 });
    traj.moveTo(state.x, state.y).lineTo(predX, predY).stroke({ color: COL.ship, width: 1, alpha: 0.22 });

    // Altitude reference: faint dashed line from ship straight down to the surface beneath.
    // Only drawn when the ship has actual headroom — avoids visual noise near touchdown.
    altLine.clear();
    const groundBelow = lossY(state.x);
    if (groundBelow - state.y > 18) {
      const dashLen = 4;
      let yCursor = state.y + 14;
      while (yCursor < groundBelow - 4) {
        altLine.moveTo(state.x, yCursor).lineTo(state.x, Math.min(yCursor + dashLen, groundBelow - 4)).stroke({ color: 0xa0aab4, width: 1, alpha: 0.45 });
        yCursor += dashLen + 4;
      }
    }

    // Thrust juice: emit small puff opposite to ship's angle every other frame.
    if (state.keys.up && state.fuel > 0) {
      if ((Math.random() < 0.55)) {
        const puffX = state.x + Math.sin(state.angle) * 14;
        const puffY = state.y + Math.cos(state.angle) * 14;
        burst(stage, puffX, puffY, COL.thrust, 1, { speed: 1.2, lifeMs: 380 });
      }
    }

    // Check bounds — off-screen is now an explicit "OFF COURSE" failure, not a silent stop.
    if (state.x < 0 || state.x > W) {
      state.over = true;
      state.reason = "off-course";
      flash(stage, 0xc44444, 240, 0.30);
      floatText(stage, Math.max(20, Math.min(W - 20, state.x)), 50, "OFF COURSE", COL.crash, { size: 18 });
    }
    
    // Landing logic
    const speed = Math.hypot(state.vx, state.vy);
    if (opts.onScoreChange) {
      opts.onScoreChange({ vram: Math.max(0, Math.floor(state.fuel)), speed: Math.floor(speed * 20) });
    }

    // Collision checking
    const groundY = lossY(state.x);

    if (state.y + 10 >= groundY) {
      state.over = true;
      state.y = groundY - 10;
      state.impactSpeed = speed;
      state.vramAtImpact = Math.max(0, Math.floor(state.fuel));

      // Impact magnitude scales every consequence — gentle landings vs crash divergence.
      const impactShake = Math.min(18, speed * 3);

      if (inGlobalPad(state.x)) {
        if (speed < maxSafeSpeed && Math.abs(state.angle) < maxSafeAngle) {
           state.won = true;
           state.reason = "win";
           // Best-soft-landing tracking. Lower is better; store as int (×100) for compare.
           const softCenti = Math.round(speed * 100);
           const prev = bestScore.get("lander-soft");
           if (!prev || softCenti < prev) {
             bestScore.set("lander-soft", softCenti);
             state.newBest = true;
             dayChip.text = bestSoftText();
           }
           // Win celebration: green burst + soft green flash + small confirming shake.
           safeBurst(stage, state.x, state.y, COL.pad, 24, { speed: 2.4, lifeMs: 900 });
           flash(stage, 0x3d9e5a, 260, 0.28);
           safeShake(stage, 4, 220);
           floatText(stage, state.x, state.y - 30, "CONVERGED!", COL.pad, { size: 24 });
        } else {
           state.reason = "diverged";
           safeBurst(stage, state.x, state.y, COL.crash, 30);
           flash(stage, 0xc44444, 220, 0.34);
           safeShake(stage, impactShake, 320);
           ship.visible = false;
           floatText(stage, state.x, state.y - 30, "DIVERGED — LR TOO HIGH", COL.crash, { size: 16 });
        }
      } else if (inLocalPad(state.x)) {
        state.reason = "local-min";
        safeBurst(stage, state.x, state.y, COL.crash, 30);
        flash(stage, 0xc87b2a, 200, 0.26);
        safeShake(stage, impactShake, 300);
        ship.visible = false;
        floatText(stage, state.x, state.y - 30, "LOCAL MINIMUM — SUBOPTIMAL", COL.crash, { size: 16 });
      } else {
        state.reason = "missed-basin";
        safeBurst(stage, state.x, state.y, COL.crash, 30);
        flash(stage, 0xc44444, 220, 0.34);
        safeShake(stage, impactShake, 320);
        ship.visible = false;
        floatText(stage, state.x, state.y - 30, "MISSED THE BASIN", COL.crash, { size: 16 });
      }
    }

    // OOM: running out of VRAM mid-air ends the run, just as it kills a real training process.
    if (state.fuel <= 0 && !state.over) {
      state.over = true;
      state.reason = "oom";
      flame.visible = false;
      safeBurst(stage, state.x, state.y, COL.crash, 26, { speed: 2.0, lifeMs: 700 });
      flash(stage, 0xc44444, 240, 0.32);
      safeShake(stage, 8, 280);
      floatText(stage, state.x, state.y - 30, "OOM — VRAM EXHAUSTED", COL.crash, { size: 16 });
    }

    if (state.over && !state.gameOverFired && opts.onGameOver) {
      state.gameOverFired = true;
      retryBtn.visible = true;
      // Recolor the retry pill to match the outcome — green confirms a win,
      // MIT-red invites another try after a loss.
      if (state.won) {
        retryBg.clear();
        retryBg.roundRect(-78, -18, 156, 36, 18).fill({ color: 0x3d9e5a }).stroke({ color: 0x256b3a, width: 1.5 });
        retryLbl.text = "↺  PLAY AGAIN";
      }
      opts.onGameOver({
        won: state.won,
        reason: state.reason,
        impactSpeed: state.impactSpeed,
        vramAtImpact: state.vramAtImpact,
        newBest: !!state.newBest,
        shareText: buildShareText(state)
      });
    }
  });

  function buildShareText(s) {
    const v = (s.impactSpeed || 0).toFixed(2);
    const vram = s.vramAtImpact ?? 0;
    const head = `🚀 Gradient Lander · Day #${day}`;
    const tail = "mlsysbook.ai/games/lander";
    if (s.reason === "win") {
      const star = s.newBest ? "  ⭐ new personal best!" : "";
      return `${head}\n🟢 CONVERGED · v=${v} · VRAM ${vram}%${star}\n${tail}`;
    }
    if (s.reason === "diverged")     return `${head}\n🔴 DIVERGED — LR too high (v=${v})\n${tail}`;
    if (s.reason === "local-min")    return `${head}\n🟠 LOCAL MIN — suboptimal solution\n${tail}`;
    if (s.reason === "off-course")   return `${head}\n🔴 OFF COURSE — weights diverged\n${tail}`;
    if (s.reason === "missed-basin") return `${head}\n🔴 MISSED THE BASIN — no clear gradient\n${tail}`;
    if (s.reason === "oom")          return `${head}\n🔴 OOM — VRAM exhausted mid-run\n${tail}`;
    return `${head}\n${tail}`;
  }

  return {
    id: "lander",
    ahaLabel: AHA.win.label,                 // legacy fallback
    ahaText: AHA.win.text,                   // legacy fallback
    aha(reason) { return AHA[reason] || AHA.win; },
    destroy() {
      window.removeEventListener('keydown', handleKeydown);
      window.removeEventListener('keyup', handleKeyup);
      app.destroy(true, { children: true, texture: true });
    }
  };
}