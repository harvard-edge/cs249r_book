/* ============================================================
   MLSysBook Playground — Straggler (v2, Pixi)
   ------------------------------------------------------------
   Pedagogical claim:
     One slow GPU stalls the whole cluster. In a ring all-reduce
     every GPU waits at the synchronization barrier; tail latency,
     not average throughput, sets the step time. Selective
     recomputation, ZeRO sharding, and gradient compression are
     all attempts to keep the slow GPU off the critical path.

   Mechanic (rhythm + response):
     - 8 anthropomorphic GPU-creatures arranged in a ring
     - Player IS GPU #0 (top of ring, 12-o'clock — "YOU")
     - The ring runs in synchronized rounds:
       * Each round = one hop of all chunks one position clockwise
       * Round duration shrinks over time (difficulty)
       * During a round: chunks animate to the next GPU
       * At end of round: if player tapped during the round window,
         the ring advances. Otherwise the ring HALTS and every GPU
         goes idle (waiting → sighing → ash)
     - 8 hops = one synchronized training step
     - Phase 2 (after 30s): adds a second wave of chunks (gather
       after scatter), doubling visual density and stake

   Visuals:
     - Vector character sprites (4 expressive states per GPU)
     - Chunks travel along the ring with a slight arc
     - GlowFilter trail on chunks (lazy-loaded)
     - "READY" beat indicator at player when tap is needed
     - Particle bursts on step completion
     - Camera shake on throttle (game over by ash)
   ============================================================ */

import {
  mountPixiOnCanvas, dailySeed, dayNumber, bestScore,
  pop, flash, burst, floatText, shake, tween, getFilters
} from "/assets/games/runtime.mjs";
import * as PIXI from "/assets/games/vendor/pixi.min.mjs";

const N_GPUS          = 8;
const PLAYER_IDX      = 0;
const TIME_LIMIT_MS   = 60000;
const PHASE2_START_MS = 30000;

/* Round timing: time in ms for one hop. Linear ramp from BASE to FAST over the
   60s game. Player must tap during the round window or the ring stalls. */
const ROUND_BASE_MS   = 2400;   // gentle start — give a fresh player rhythm
const ROUND_FAST_MS   = 700;    // by minute mark, demands precision
const GRACE_PERIOD_MS = 4000;   // first 4s: ring runs but DOESN'T penalize misses

/* Stall timing: how long the ring can be stalled before player throttles. */
const STALL_WAIT_MS   = 700;    // → "waiting" state on player + others
const STALL_SIGH_MS   = 1800;   // → "sighing"
const STALL_ASH_MS    = 4500;   // → ash, game over (extra forgiving)

const COL = {
  bg:           0xfafafa,
  ringStroke:   0xc8d3dc,
  ringDot:      0x9ab0bf,
  ringStrokeStall: 0xe2a23a,
  text:         0x333333,
  muted:        0x888888,
  faint:        0xeeeeee,
  blue:         0x4a90c4,
  blueLight:    0xcfe2f3,
  green:        0x3d9e5a,
  greenLight:   0xc9e4cc,
  red:          0xc44444,
  redLight:     0xf9d6d5,
  orange:       0xc87b2a,
  orangeLight:  0xfdebd0,
  purple:       0x6a4a7a,
  purpleLight:  0xe1d5e7,
  mitRed:       0xa31f34,
  white:        0xffffff,
  ash:          0x9aa0a8,
};

const SPRITES = {
  idle:    "/assets/games/sprites/gpu-idle.png",
  waiting: "/assets/games/sprites/gpu-waiting.png",
  sighing: "/assets/games/sprites/gpu-sighing.png",
  ash:     "/assets/games/sprites/gpu-ash.png",
  chunk:   "/assets/games/sprites/chunk.png",
};

window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};
window.MLSP.games.straggler = function (canvas, opts) { return mountStraggler(canvas, opts); };

export async function mountStraggler(canvas, opts = {}) {
  const { app, stage, width: W, height: H } = await mountPixiOnCanvas(canvas, { bg: COL.bg });
  const { rand, today } = dailySeed("straggler");

  /* --- Lazy-load filters for the chunk glow trail --- */
  let filters = null;
  getFilters().then(f => { filters = f; }).catch(() => { /* graceful fallback */ });

  /* --- Load sprite textures --- */
  const textures = await Promise.all([
    PIXI.Assets.load(SPRITES.idle),
    PIXI.Assets.load(SPRITES.waiting),
    PIXI.Assets.load(SPRITES.sighing),
    PIXI.Assets.load(SPRITES.ash),
    PIXI.Assets.load(SPRITES.chunk),
  ]);
  const TEX = {
    idle: textures[0], waiting: textures[1], sighing: textures[2],
    ash: textures[3], chunk: textures[4],
  };

  /* --- Geometry --- */
  const cx = W / 2, cy = H / 2 + 10;
  const ringR = Math.min(W, H) * 0.34;
  const gpuPositions = [];
  for (let i = 0; i < N_GPUS; i++) {
    const ang = -Math.PI / 2 + i * (2 * Math.PI / N_GPUS);
    gpuPositions.push({ x: cx + ringR * Math.cos(ang), y: cy + ringR * Math.sin(ang), ang });
  }

  /* --- Game state ---
     RING SEMANTICS:
       hop:        the integer hop count this round-set (0..N_GPUS-1)
       roundT:     animation progress for the current hop (0..1)
       roundMs:    duration of one hop at current difficulty
       waitingFor: true ↔ ring is paused at end of a hop, awaiting player tap
       playerReady: true ↔ player tapped during current/previous round
       stallMs:    how long we have been waiting in the current stall
   */
  const state = {
    timeLeft: TIME_LIMIT_MS,
    elapsed: 0,
    over: false,
    endReason: null,
    /* Ring */
    hopWithinStep: 0,
    roundT: 0,
    waitingFor: false,
    playerReady: false,
    stallMs: 0,
    /* Phase 2 */
    phase2Active: false,
    /* Score */
    stepsCompleted: 0,
    badRounds: 0,           // rounds where player wasn't ready in time
    /* Idle accounting (the "shame" metric) */
    idleAccumGpuMs: 0,
    totalGpuMs: 0,
  };
  const alltimeBestRef = { v: bestScore.get("straggler") };

  /* --- Layers --- */
  const bgLayer = new PIXI.Container();
  const ringLayer = new PIXI.Container();
  const gpuLayer = new PIXI.Container();
  const chunkLayer = new PIXI.Container();   // chunks render ABOVE GPUs so they stay visible
  const labelLayer = new PIXI.Container();
  const overlayLayer = new PIXI.Container();
  stage.addChild(bgLayer, ringLayer, gpuLayer, chunkLayer, labelLayer, overlayLayer);

  /* --- Background ring --- */
  const ringGfx = new PIXI.Graphics();
  function drawRing(stalled = false) {
    ringGfx.clear();
    ringGfx.circle(cx, cy, ringR);
    ringGfx.stroke({ width: stalled ? 3 : 2, color: stalled ? COL.ringStrokeStall : COL.ringStroke, alpha: stalled ? 0.85 : 0.7 });
    for (let i = 0; i < N_GPUS; i++) {
      const a1 = gpuPositions[i].ang;
      const a2 = gpuPositions[(i + 1) % N_GPUS].ang;
      let span = a2 - a1;
      /* Avoid wrap discontinuity for the last segment */
      if (span < 0) span += 2 * Math.PI;
      for (let k = 1; k <= 3; k++) {
        const a = a1 + span * (k / 4);
        const dx = cx + ringR * Math.cos(a);
        const dy = cy + ringR * Math.sin(a);
        ringGfx.circle(dx, dy, 1.4);
        ringGfx.fill({ color: COL.ringDot, alpha: 0.5 });
      }
    }
  }
  ringLayer.addChild(ringGfx);
  drawRing();

  /* --- Spawn 8 GPU sprites --- */
  const gpus = [];
  for (let i = 0; i < N_GPUS; i++) {
    const sp = new PIXI.Sprite(TEX.idle);
    sp.anchor.set(0.5, 0.55);
    sp.position.set(gpuPositions[i].x, gpuPositions[i].y);
    const targetW = 88;
    sp.scale.set(targetW / sp.texture.width);
    if (i === PLAYER_IDX) sp.scale.set((targetW * 1.2) / sp.texture.width);
    sp.eventMode = "static";
    sp.cursor = i === PLAYER_IDX ? "pointer" : "default";
    gpuLayer.addChild(sp);
    gpus.push({ idx: i, sprite: sp, stateName: "idle", isPlayer: i === PLAYER_IDX });
  }

  /* "YOU" label below player GPU */
  const youLabel = new PIXI.Text({
    text: "YOU",
    style: {
      fontFamily: "Helvetica Neue, Arial, sans-serif",
      fontSize: 13, fontWeight: "800", fill: COL.mitRed, letterSpacing: 1.4
    }
  });
  youLabel.anchor.set(0.5, 0);
  youLabel.position.set(gpuPositions[PLAYER_IDX].x, gpuPositions[PLAYER_IDX].y - 80);
  labelLayer.addChild(youLabel);

  /* GPU index labels (#0..#7) */
  for (let i = 0; i < N_GPUS; i++) {
    const lbl = new PIXI.Text({
      text: "#" + i,
      style: {
        fontFamily: "Helvetica Neue, Arial, sans-serif",
        fontSize: 10, fill: COL.muted, letterSpacing: 0.4
      }
    });
    lbl.anchor.set(0.5, 0);
    const ax = gpuPositions[i].x - cx, ay = gpuPositions[i].y - cy;
    const aL = Math.hypot(ax, ay);
    lbl.position.set(gpuPositions[i].x + (ax / aL) * 56, gpuPositions[i].y + (ay / aL) * 56);
    labelLayer.addChild(lbl);
  }

  /* --- "READY" beat indicator: pulse around the player when input is needed --- */
  const beatGfx = new PIXI.Graphics();
  overlayLayer.addChild(beatGfx);
  function drawBeat(intensity) {
    /* intensity 0..1 — drives radius and alpha */
    beatGfx.clear();
    if (intensity <= 0) return;
    const r = 50 + intensity * 12;
    beatGfx.circle(gpuPositions[PLAYER_IDX].x, gpuPositions[PLAYER_IDX].y, r);
    beatGfx.stroke({ width: 3, color: COL.mitRed, alpha: 0.55 * intensity });
  }

  /* --- Tutorial banner: "TAP IN RHYTHM" during grace period --- */
  const tutorialText = new PIXI.Text({
    text: "TAP IN RHYTHM  ·  press SPACE or click YOUR GPU",
    style: {
      fontFamily: "Helvetica Neue, Arial, sans-serif",
      fontSize: 14, fontWeight: "700",
      fill: COL.mitRed, letterSpacing: 1.4
    }
  });
  tutorialText.anchor.set(0.5, 0);
  tutorialText.position.set(cx, H - 70);
  overlayLayer.addChild(tutorialText);

  /* --- Helpers --- */
  function setGpuState(g, name) {
    if (g.stateName === name) return;
    g.stateName = name;
    g.sprite.texture = TEX[name] || TEX.idle;
  }
  function isIdle(g) {
    return g.stateName === "waiting" || g.stateName === "sighing" || g.stateName === "ash";
  }

  /* --- Chunk objects: one per GPU, arranged synchronously around the ring.
     Each has: ownerIdx (which GPU originally owned it — fixed), phase, sprite.
     Their position is derived from `state.hopWithinStep + state.roundT` plus their
     ownerIdx offset around the ring. */
  const chunks = [];
  function spawnChunkSet(phase) {
    for (let i = 0; i < N_GPUS; i++) {
      const sp = new PIXI.Sprite(TEX.chunk);
      sp.anchor.set(0.5, 0.5);
      const targetW = phase === 1 ? 38 : 30;
      sp.scale.set(targetW / sp.texture.width);
      if (phase === 2) sp.tint = 0xfff1c2;
      chunkLayer.addChild(sp);
      if (filters && filters.GlowFilter) {
        sp.filters = [new filters.GlowFilter({
          distance: 14, outerStrength: phase === 2 ? 1.6 : 1.2,
          innerStrength: 0, color: phase === 2 ? 0xf6c177 : 0x3d9e5a, quality: 0.4
        })];
      }
      chunks.push({ ownerIdx: i, phase, sprite: sp });
    }
  }
  spawnChunkSet(1);

  /* Place a chunk at fractional position around the ring.
     posF: a real number, where integer = GPU index, fractional = arc to next.
     For phase 2 chunks we offset their ring-position by N/2 so they're on the
     opposite side — visually parallel to phase 1. */
  function placeChunk(c) {
    /* c.ownerIdx is the chunk's "home". As the ring rotates, the chunk's
       physical location shifts clockwise by hopWithinStep + roundT. */
    let posF = c.ownerIdx + state.hopWithinStep + state.roundT;
    if (c.phase === 2) posF += N_GPUS / 2;   // offset for the gather wave
    const fromIdx = Math.floor(posF) % N_GPUS;
    const toIdx = (fromIdx + 1) % N_GPUS;
    const t = posF - Math.floor(posF);
    const fromPos = gpuPositions[fromIdx];
    const toPos = gpuPositions[toIdx];
    const px = fromPos.x + (toPos.x - fromPos.x) * t;
    const py = fromPos.y + (toPos.y - fromPos.y) * t;
    /* Pull inward (toward centre) along the arc — gives the chunk a curved
       path that suggests "passed across the ring", and avoids overlap with
       the GPU sprites themselves. */
    const midPullX = (cx - (fromPos.x + toPos.x) / 2) * 0.20;
    const midPullY = (cy - (fromPos.y + toPos.y) / 2) * 0.20;
    const arcK = 4 * t * (1 - t);
    /* Offset radially OUTWARD so chunks are visible in the ring's gap zone
       outside the GPU sprites. Phase-2 chunks are pushed even further out. */
    const offsetR = c.phase === 2 ? 56 : 38;
    const dx = px - cx, dy = py - cy, dL = Math.hypot(dx, dy) || 1;
    const ox = (dx / dL) * offsetR, oy = (dy / dL) * offsetR;
    c.sprite.position.set(px + midPullX * arcK + ox, py + midPullY * arcK + oy);
    /* Gentle bobble */
    c.sprite.rotation = Math.sin(state.elapsed * 0.005 + c.ownerIdx) * 0.10;
  }

  /* --- Input handling --- */
  function tap() {
    if (state.over) return;
    if (state.playerReady) {
      /* Already armed for this round — gentle confirm */
      pop(overlayLayer, gpuPositions[PLAYER_IDX].x, gpuPositions[PLAYER_IDX].y, COL.muted, { r: 18, ms: 220 });
      return;
    }
    state.playerReady = true;
    /* Snappy green pop on the player */
    pop(overlayLayer, gpuPositions[PLAYER_IDX].x, gpuPositions[PLAYER_IDX].y, COL.green, { r: 26, ms: 320 });
    floatText(overlayLayer, gpuPositions[PLAYER_IDX].x, gpuPositions[PLAYER_IDX].y - 95, "READY", COL.green, { size: 13, lifeMs: 600 });
    /* If the ring is currently STALLED, releasing now snaps it forward */
    if (state.waitingFor) {
      advanceHop();
    }
  }

  gpus[PLAYER_IDX].sprite.on("pointertap", tap);
  app.stage.on("pointertap", (e) => {
    if (e.target === gpus[PLAYER_IDX].sprite) return;
    tap();
  });
  function keyHandler(e) {
    if (e.code === "Space") { e.preventDefault(); tap(); }
    else if (e.key === "r" || e.key === "R") { if (opts.onRetry) opts.onRetry(); }
  }
  window.addEventListener("keydown", keyHandler);

  /* --- Step / hop transitions --- */
  function advanceHop() {
    /* Move forward by one position. Reset round state. */
    state.hopWithinStep = (state.hopWithinStep + 1) % N_GPUS;
    state.roundT = 0;
    state.waitingFor = false;
    state.playerReady = false;
    state.stallMs = 0;
    drawRing(false);
    /* All non-player GPUs return to idle on a successful hop */
    for (const g of gpus) setGpuState(g, "idle");
    /* Step complete? */
    if (state.hopWithinStep === 0) {
      state.stepsCompleted++;
      flash(stage, COL.green, 240, 0.16);
      for (let i = 0; i < N_GPUS; i++) {
        burst(overlayLayer, gpuPositions[i].x, gpuPositions[i].y, COL.green, 14, { speed: 3, lifeMs: 700 });
      }
      floatText(overlayLayer, cx, cy, "STEP " + state.stepsCompleted + " ✓", COL.green, { size: 22, lifeMs: 900 });
    }
  }

  function endGame(reason) {
    if (state.over) return;
    state.over = true;
    state.endReason = reason;
    if (state.stepsCompleted > alltimeBestRef.v) {
      alltimeBestRef.v = state.stepsCompleted;
      bestScore.set("straggler", state.stepsCompleted);
    }
    if (reason === "throttled") {
      flash(stage, COL.mitRed, 480, 0.4);
      shake(gpuLayer, 18, 700);
      for (const g of gpus) setGpuState(g, "ash");
    } else {
      flash(stage, COL.green, 320, 0.18);
    }
    const txt = reason === "throttled"
      ? "THROTTLED — cluster melted down"
      : "TIME — " + state.stepsCompleted + " steps, " + Math.round(idlePct()) + "% idle";
    const t = new PIXI.Text({
      text: txt,
      style: {
        fontFamily: "Helvetica Neue, Arial, sans-serif",
        fontSize: 18, fontWeight: "800",
        fill: reason === "throttled" ? COL.mitRed : COL.text,
        align: "center"
      }
    });
    t.anchor.set(0.5, 0.5);
    t.position.set(cx, H - 38);
    overlayLayer.addChild(t);

    if (opts.onGameOver) {
      opts.onGameOver({
        stepsCompleted: state.stepsCompleted,
        idlePct: idlePct(),
        endReason: state.endReason,
        emojiGrid: buildEmojiGrid(),
        alltimeBest: alltimeBestRef.v,
      });
    }
  }

  function idlePct() {
    if (state.totalGpuMs <= 0) return 0;
    return 100 * (state.idleAccumGpuMs / state.totalGpuMs);
  }

  function buildEmojiGrid() {
    const idle = idlePct();
    const steps = state.stepsCompleted;
    let row1, row2, row3;
    if (state.endReason === "throttled") {
      row1 = "💀💀💀💀💀💀💀💀";
      row2 = "🔥🔥🔥🔥🔥🔥🔥🔥";
      row3 = "________________";
    } else {
      const goodHops = Math.min(8, Math.round(8 * (1 - idle / 100)));
      row1 = "🟢".repeat(goodHops) + "⬜".repeat(8 - goodHops);
      row2 = (steps >= 8 ? "⚡⚡⚡⚡⚡⚡⚡⚡" : "⚡".repeat(steps) + "·".repeat(Math.max(0, 8 - steps)));
      row3 = "🤖🤖🤖🤖🤖🤖🤖🤖";
    }
    return row1 + "\n" + row2 + "\n" + row3;
  }

  /* --- Frame loop --- */
  app.ticker.add((ticker) => {
    const dt = ticker.deltaMS;
    if (state.over) return;

    state.timeLeft -= dt;
    state.elapsed += dt;
    state.totalGpuMs += dt * N_GPUS;

    /* Phase 2 trigger */
    if (!state.phase2Active && state.elapsed >= PHASE2_START_MS) {
      state.phase2Active = true;
      flash(stage, COL.orange, 320, 0.16);
      floatText(overlayLayer, cx, cy - ringR - 22, "PHASE 2: gather pass", COL.orange, { size: 14, lifeMs: 1400 });
      spawnChunkSet(2);
    }

    /* Round duration ramps from BASE → FAST linearly with elapsed time */
    const tFrac = Math.min(1, state.elapsed / TIME_LIMIT_MS);
    const roundMs = ROUND_BASE_MS + (ROUND_FAST_MS - ROUND_BASE_MS) * tFrac;

    /* Fade out tutorial banner over the second half of the grace period */
    if (state.elapsed < GRACE_PERIOD_MS) {
      const fadeStart = GRACE_PERIOD_MS * 0.5;
      if (state.elapsed > fadeStart) {
        tutorialText.alpha = 1 - (state.elapsed - fadeStart) / (GRACE_PERIOD_MS - fadeStart);
      }
    } else if (tutorialText.alpha > 0) {
      tutorialText.alpha = 0;
    }

    /* Advance the round timer (only if not stalled) */
    if (!state.waitingFor) {
      state.roundT += dt / roundMs;
      if (state.roundT >= 1) {
        state.roundT = 1;
        /* Ring wants to advance. Did the player tap?
           During grace period, auto-pass to teach the rhythm without punishing. */
        const inGrace = state.elapsed < GRACE_PERIOD_MS;
        if (state.playerReady || inGrace) {
          advanceHop();
        } else {
          /* Stall begins. Don't change GPU sprites yet — they'll progress
             through waiting → sighing → ash as stall time accumulates. */
          state.waitingFor = true;
          state.stallMs = 0;
          drawRing(true);
        }
      }
    } else {
      /* We are stalled. Accumulate stall time. */
      state.stallMs += dt;
      /* Other GPUs progressively look more impatient/exhausted */
      let blockedState = "idle";
      if (state.stallMs >= STALL_SIGH_MS) blockedState = "sighing";
      else if (state.stallMs >= STALL_WAIT_MS) blockedState = "waiting";
      for (let i = 0; i < N_GPUS; i++) {
        if (i === PLAYER_IDX) continue;
        setGpuState(gpus[i], blockedState);
      }
      /* Player's own state escalates too */
      if (state.stallMs >= STALL_ASH_MS) {
        setGpuState(gpus[PLAYER_IDX], "ash");
        endGame("throttled");
        return;
      } else if (state.stallMs >= STALL_SIGH_MS) {
        setGpuState(gpus[PLAYER_IDX], "sighing");
      } else if (state.stallMs >= STALL_WAIT_MS) {
        setGpuState(gpus[PLAYER_IDX], "waiting");
      }
      state.badRounds = state.badRounds; // no-op, tracked elsewhere
    }

    /* --- Position chunks --- */
    for (const c of chunks) placeChunk(c);

    /* --- Beat indicator: pulse strength ramps with how close we are to needing input ---
       During roundT in [0, 0.6]: no beat
       [0.6, 1.0]: ramp up to encourage tap
       Stalled: full strong pulse */
    let beatI = 0;
    if (state.waitingFor) {
      beatI = 0.7 + 0.3 * Math.sin(state.elapsed * 0.012);
    } else if (state.roundT > 0.6 && !state.playerReady) {
      beatI = (state.roundT - 0.6) / 0.4;
    } else if (state.playerReady) {
      beatI = 0;   // armed, no pulse needed
    }
    drawBeat(beatI);

    /* --- Idle accounting (the shame metric) ---
       Sum (number of non-idle-state GPUs) * dt across all GPUs, divided
       by total GPU-ms, gives the cluster idle %. */
    for (const g of gpus) {
      if (isIdle(g)) state.idleAccumGpuMs += dt;
    }

    /* Time up */
    if (state.timeLeft <= 0) {
      state.timeLeft = 0;
      endGame("time");
      return;
    }

    /* HUD callback */
    if (opts.onScoreChange) {
      opts.onScoreChange({
        stepsCompleted: state.stepsCompleted,
        idlePct: idlePct(),
        timeLeft: state.timeLeft,
        alltimeBest: alltimeBestRef.v,
      });
    }
  });

  /* --- Cleanup --- */
  function cleanup() {
    window.removeEventListener("keydown", keyHandler);
  }

  /* --- Aha-card / share API --- */
  const api = {
    ahaLabel: "Aha:",
    ahaText: "One slow GPU stalls the cluster. In ring all-reduce every worker waits at the barrier — tail latency, not average throughput, sets the step time. That's why distributed training fights with overlap (gradient compression, ZeRO sharding, pipeline bubbles, redundant computation): keep the slow one off the critical path.",
    ahaLink: { href: "https://eng.uber.com/horovod/", label: "Horovod & ring all-reduce →" },
    buildShareText(r) {
      return "MLSysBook Playground · Straggler · Day " + (window.MLSP.dayNumber ? window.MLSP.dayNumber() : today) + "\n" +
        (r.endReason === "throttled" ? "THROTTLED 💀\n" : (r.stepsCompleted + " steps, " + Math.round(r.idlePct) + "% cluster idle\n")) +
        r.emojiGrid + "\n" +
        "play → mlsysbook.ai/games/straggler/";
    },
    _state: state,
    _cleanup: cleanup,
  };

  return api;
}
