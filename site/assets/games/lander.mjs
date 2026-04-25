import { mountPixiOnCanvas, burst, floatText, flash, shake } from "/assets/games/runtime.mjs";
import * as P from "/assets/games/vendor/pixi.min.mjs";

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
    keys: { up: false, left: false, right: false }
  };

  const gravity = 0.05;
  const thrustPower = 0.15;
  const rotSpeed = 0.055;            // tuned down from 0.08 — easier precision, still feels responsive
  const maxSafeSpeed = 2.4;          // tuned up from 2.0 — first-time forgiveness, still teaches the lesson
  const maxSafeAngle = 0.6;          // tuned up from 0.5 (~34°) — same reasoning
  const terrain = createTerrain();

  function rand(min, max) {
    return min + Math.random() * (max - min);
  }

  function createTerrain() {
    const globalX = rand(W * 0.42, W * 0.58);
    const leftX = rand(W * 0.17, W * 0.32);
    const rightX = rand(W * 0.68, W * 0.83);
    return {
      phase: rand(0, Math.PI * 2),
      slope: rand(-4, 4),  // tuned down from rand(-8, 8) — caps random unfairness from extreme tilts
      global: { x: globalX, width: rand(86, 112), wellWidth: rand(72, 92), depth: rand(50, 66) },
      locals: [
        { x: leftX, width: rand(70, 92), wellWidth: rand(54, 70), depth: rand(24, 36) },
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
    const base = H - 96
      + terrain.slope * (normalized - 0.5)
      + 13 * Math.sin(normalized * Math.PI * 2.1 + terrain.phase)
      + 7 * Math.sin(normalized * Math.PI * 4.7 + terrain.phase * 0.53);
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

  const globalLabel = new P.Text({ text: "global minimum", style: { fill: 0x3d9e5a, fontSize: 11, fontWeight: "bold" }});
  globalLabel.anchor.set(0.5);
  globalLabel.position.set(terrain.global.x, lossY(terrain.global.x) - 18);
  const localLabel = new P.Text({ text: "local minima", style: { fill: 0x9a6620, fontSize: 10 }});
  localLabel.anchor.set(0.5);
  localLabel.position.set(terrain.locals[0].x, lossY(terrain.locals[0].x) - 17);

  stage.addChild(bg, contourLayer, basinWash, surface, padGlobal, padLocal1, padLocal2, globalLabel, localLabel);

  // Predictive trajectory marker — translucent target showing where the ship will be in
  // ~0.5 s if the player keeps coasting (no thrust). Helps a new player anticipate the basin.
  const traj = new P.Graphics();
  stage.addChild(traj);

  // Ship
  const ship = new P.Graphics();
  ship.moveTo(0, -15).lineTo(10, 10).lineTo(-10, 10).lineTo(0, -15).fill({color: COL.ship});
  stage.addChild(ship);

  // Flame
  const flame = new P.Graphics();
  flame.moveTo(-5, 10).lineTo(0, 25).lineTo(5, 10).fill({color: COL.thrust});
  flame.visible = false;
  ship.addChild(flame);

  const noiseText = new P.Text({ text: "stochastic gradient noise", style: { fill: 0x777777, fontSize: 13 }});
  noiseText.position.set(18, 18);
  stage.addChild(noiseText);

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
    text: "↑  thrust  (burns VRAM)\n← →  steer learning rate",
    style: { fill: 0xffffff, fontSize: 15, align: "center", lineHeight: 22 }
  });
  readyControls.anchor.set(0.5);
  readyControls.position.set(W / 2, H / 2 + 14);
  ready.addChild(readyControls);
  const readyCta = new P.Text({ text: "▲  press UP to launch", style: { fill: 0xffd6a8, fontSize: 18, fontWeight: "700" }});
  readyCta.anchor.set(0.5);
  readyCta.position.set(W / 2, H / 2 + 70);
  ready.addChild(readyCta);
  stage.addChild(ready);
  let ctaPulseT = 0;

  const handleKeydown = (e) => {
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      state.keys.up = true;
      if (!state.started) {
        state.started = true;
        ready.visible = false;
      }
    }
    if (e.key === 'ArrowLeft') { e.preventDefault(); state.keys.left = true; }
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

  app.ticker.add(() => {
    // HUD drawn every frame so it never goes stale between win/crash and the aha card.
    drawHud();

    // Goal pad always pulses (so the player's eye finds it before launch and during play).
    pulseT += 0.06;
    const r = 26 + Math.sin(pulseT) * 6;
    padPulse.clear();
    padPulse.circle(0, 0, r).stroke({ color: COL.pad, width: 2, alpha: 0.55 });
    padPulse.circle(0, 0, r * 0.55).stroke({ color: COL.pad, width: 1.4, alpha: 0.35 });

    if (!state.started) {
      // Pre-game: pulse the "press UP" prompt so it reads as an action, not a static label.
      ctaPulseT += 0.08;
      readyCta.alpha = 0.75 + 0.25 * Math.sin(ctaPulseT);
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
      
      // Impact magnitude scales every consequence — gentle landings vs crash divergence.
      const impactShake = Math.min(18, speed * 3);

      if (inGlobalPad(state.x)) {
        if (speed < maxSafeSpeed && Math.abs(state.angle) < maxSafeAngle) {
           state.won = true;
           // Win celebration: green burst + soft green flash + small confirming shake.
           burst(stage, state.x, state.y, COL.pad, 24, { speed: 2.4, lifeMs: 900 });
           flash(stage, 0x3d9e5a, 260, 0.28);
           shake(stage, 4, 220);
           floatText(stage, state.x, state.y - 30, "CONVERGED!", COL.pad, { size: 24 });
        } else {
           burst(stage, state.x, state.y, COL.crash, 30);
           flash(stage, 0xc44444, 220, 0.34);
           shake(stage, impactShake, 320);
           ship.visible = false;
           floatText(stage, state.x, state.y - 30, "DIVERGED (TOO FAST)", COL.crash, { size: 16 });
        }
      } else if (inLocalPad(state.x)) {
        burst(stage, state.x, state.y, COL.crash, 30);
        flash(stage, 0xc87b2a, 200, 0.26);
        shake(stage, impactShake, 300);
        ship.visible = false;
        floatText(stage, state.x, state.y - 30, "LOCAL MINIMUM (Suboptimal)", COL.crash, { size: 16 });
      } else {
        burst(stage, state.x, state.y, COL.crash, 30);
        flash(stage, 0xc44444, 220, 0.34);
        shake(stage, impactShake, 320);
        ship.visible = false;
        floatText(stage, state.x, state.y - 30, "GRADIENT EXPLOSION (Crash)", COL.crash, { size: 16 });
      }
    }

    if (state.fuel <= 0 && flame.visible) {
      flame.visible = false;
      floatText(stage, state.x, state.y, "OOM!", COL.crash, {size: 14});
    }

    if (state.over && opts.onGameOver) {
      opts.onGameOver({
         won: state.won
      });
    }
  });

  return {
    id: "lander",
    ahaLabel: "You just experienced",
    ahaText: "Training with a massive batch size gives you a perfectly stable 'thrust' down the loss landscape, but it consumes your VRAM extremely fast. Finding the right balance between Batch Size (fuel burn rate) and Learning Rate (steering angle) is the only way to land in the global minimum without diverging or running out of memory.",
    destroy() {
      window.removeEventListener('keydown', handleKeydown);
      window.removeEventListener('keyup', handleKeyup);
      app.destroy(true, { children: true, texture: true });
    }
  };
}