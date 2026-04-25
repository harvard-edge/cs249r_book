/* ============================================================
   MLSysBook Playground — Quantization Sharp Shot (v2, Pixi visual lift)
   ------------------------------------------------------------
   Mechanic preserved from v1:
     - 6 per-layer precision dials (fp32 / fp16 / int8 / int4)
     - 96-bit budget; over-budget = no shot
     - 10 shots per round
     - Edge layers drift on int4, attn jitters, ffn blurs
     - On miss, briefly reveal true target

   v2 changes (visual + pedagogy):
     - Concentric scoring zones (bullseye 3, mid 2, outer 1, miss 0).
       Win threshold: 18 / 30 to ship. Reframes hit/miss as a gradient
       — close shots count, perfect shots reward, wild shots zero.
     - LIVE accuracy estimate updates as you cycle dials. Lets you see
       "this config is over-budget AND only 67% accurate" before firing.
     - Per-layer sensitivity hints displayed inline ("edge — high",
       "attn — med", "ffn — low") so the skill ceiling is "memorise
       sensitivity, allocate cleverly", not "guess".
     - Pixi-drawn animated bullseye with concentric rings, centre glow,
       slow rotation on outer ring.
     - BlurFilter applied directly to target when ffn blur is high.
     - Crosshair pulses (red over-budget, green good-config).
     - Shot trail: line from crosshair to landing + impact burst.
     - Layer dial tweens between stops; dial colour ramps green→red
       with sensitivity to telegraph cost.
   ============================================================ */

import {
  mountPixiOnCanvas, dailySeed, dayNumber, bestScore,
  pop, flash, burst, floatText, shake, tween, getFilters
} from "./runtime.mjs";
import * as PIXI from "./vendor/pixi.min.mjs";

const PRECISIONS = [
  { name: "fp32", bits: 32, color: 0xcfe2f3, stroke: 0x4a90c4 },
  { name: "fp16", bits: 16, color: 0xd4edda, stroke: 0x3d9e5a },
  { name: "int8", bits: 8,  color: 0xfdebd0, stroke: 0xc87b2a },
  { name: "int4", bits: 4,  color: 0xf9d6d5, stroke: 0xc44444 }
];
const NUM_LAYERS       = 6;
const MAX_BUDGET       = 96;
const SHOTS_PER_ROUND  = 10;
const TARGET_RADIUS    = 52;     // outermost ring (worth +1) — bigger for visual presence
const MID_RADIUS       = 32;     // worth +2
const BULLSEYE_RADIUS  = 14;     // worth +3
const SHIP_THRESHOLD   = 18;     // 60% of theoretical max 30

/* Layer roles drive HOW low precision degrades vision.
   "edge" = embedding/output, the sensitive ones.
   "attn" = attention layers, amplify noise.
   "ffn"  = feedforward, robust.                                   */
const LAYER_ROLES = ["edge", "attn", "ffn", "attn", "ffn", "edge"];
const LAYER_NAMES = ["embedding", "attn.1", "ffn.1", "attn.2", "ffn.2", "output"];
const SENSITIVITY = { edge: "high", attn: "med", ffn: "low" };

const COL = {
  bg:           0xfafafa,
  text:         0x333333,
  muted:        0x888888,
  faint:        0xeeeeee,
  panelBg:      0xf3f5f7,
  panelStroke:  0xd6dde3,
  budgetGood:   0x4a90c4,
  budgetWarn:   0xc87b2a,
  budgetOver:   0xc44444,
  fireGood:     0xa31f34,
  fireBad:      0xbbbbbb,
  hitGreen:     0x3d9e5a,
  missRed:      0xc44444,
  ringOuter:    0xfdebd0,
  ringMid:      0xcfe2f3,
  ringInner:    0xf9d6d5,
  bullseye:     0xa31f34,
  white:        0xffffff
};

/* Register on the legacy window.MLSP API so the .qmd boot script
   can call MLSP.games.quantization(canvas, opts).             */
window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};
window.MLSP.games.quantization = function (canvas, opts) {
  return mountSharpShot(canvas, opts);
};

export async function mountSharpShot(canvas, opts = {}) {
  const { app, stage, width: W, height: H, PIXI: P } =
    await mountPixiOnCanvas(canvas, { bg: COL.bg });
  const { rand, today } = dailySeed("quantization");
  const alltimeBestRef = { v: bestScore.get("quantization") };

  /* --- Layout --- */
  /* Left panel: layer dials + budget + accuracy + fire button (260 px wide).
     Right area: target zone (rest of canvas).                            */
  const PANEL_W = 270;
  const PANEL_X = 12;
  const PANEL_Y = 56;
  const TARGET = {
    x: PANEL_W + 24,
    y: 60,
    w: W - PANEL_W - 36,
    h: H - 110
  };
  TARGET.cx = TARGET.x + TARGET.w / 2;
  TARGET.cy = TARGET.y + TARGET.h / 2;

  /* --- State --- */
  const state = {
    layers: Array.from({ length: NUM_LAYERS }, () => ({
      precisionIdx: 0,           // start at fp32 → way over budget; player must reduce
      animatedIdx: 0             // tweened display value
    })),
    crosshairX: TARGET.cx,
    crosshairY: TARGET.cy,
    trueTargetX: TARGET.cx,
    trueTargetY: TARGET.cy,
    targetPhase: 0,
    shotsLeft:   SHOTS_PER_ROUND,
    score:       0,              // sum of zone points
    perShotZones: [],            // 0/1/2/3 per shot for emoji grid
    over:        false,
    won:         false,
    lastShot:    null,           // { x, y, hit, zone, ms }
    shotFlashMs: 0,
    revealMs:    0,
    /* derived from layers — set by recomputeEffects */
    blur: 0, jitter: 0, driftX: 0, driftY: 0, accuracy: 100,
    /* tutorial */
    showTutorial: true
  };

  /* Move true target to a fresh random in-bounds position. */
  function movetrueTarget() {
    const padding = 60;
    state.trueTargetX = TARGET.x + padding + rand() * (TARGET.w - 2 * padding);
    state.trueTargetY = TARGET.y + padding + rand() * (TARGET.h - 2 * padding);
  }
  movetrueTarget();

  /* --- Stage layers (z-order) --- */
  const bgLayer       = new P.Container();
  const targetLayer   = new P.Container();
  const targetWrap    = new P.Container();   // wraps target so we can apply BlurFilter
  targetLayer.addChild(targetWrap);
  const overlayLayer  = new P.Container();
  const panelLayer    = new P.Container();
  const hudLayer      = new P.Container();
  stage.addChild(bgLayer, targetLayer, overlayLayer, panelLayer, hudLayer);

  /* --- Filters --- */
  let filters = null;
  getFilters().then(mod => { filters = mod; applyFilters(); });

  function applyFilters() {
    if (!filters) return;
    /* No filters at start — they get applied in updateEffects() based on jitter/blur */
  }

  /* --- HUD: title + legend --- */
  const titleText = new P.Text({
    text: "Quantization Sharp Shot",
    style: { fontFamily: "Helvetica Neue, Arial", fontSize: 15, fontWeight: "700", fill: COL.text }
  });
  titleText.anchor.set(0.5, 0); titleText.position.set(W / 2, 14);
  hudLayer.addChild(titleText);

  const subText = new P.Text({
    text: "click a layer to cycle precision · lower bits = blur / jitter / drift · " +
          SHIP_THRESHOLD + " pts to ship",
    style: { fontFamily: "Helvetica Neue, Arial", fontSize: 10.5, fill: COL.muted }
  });
  subText.anchor.set(0.5, 0); subText.position.set(W / 2, 34);
  hudLayer.addChild(subText);

  /* --- Left panel background --- */
  const panelBg = new P.Graphics();
  panelBg.roundRect(PANEL_X, PANEL_Y, PANEL_W, H - PANEL_Y - 20, 8)
         .fill({ color: COL.panelBg })
         .stroke({ color: COL.panelStroke, width: 1 });
  panelLayer.addChild(panelBg);

  /* --- Layer dials --- */
  const dialRows = [];
  const DIAL_PAD = 14;
  const DIAL_H   = 44;
  const DIAL_X   = PANEL_X + DIAL_PAD;
  const DIAL_W   = PANEL_W - 2 * DIAL_PAD;
  const DIALS_Y  = PANEL_Y + 12;

  for (let i = 0; i < NUM_LAYERS; i++) {
    const rowY = DIALS_Y + i * DIAL_H;
    const row = new P.Container();
    row.position.set(DIAL_X, rowY);

    /* Background pill */
    const pill = new P.Graphics();
    row.addChild(pill);

    /* Layer name + role hint */
    const nameTxt = new P.Text({
      text: LAYER_NAMES[i],
      style: { fontFamily: "Helvetica Neue, Arial", fontSize: 11.5, fontWeight: "700", fill: COL.text }
    });
    nameTxt.position.set(8, 4);
    row.addChild(nameTxt);

    const role = LAYER_ROLES[i];
    const sensitivityColor =
      SENSITIVITY[role] === "high" ? COL.budgetOver :
      SENSITIVITY[role] === "med"  ? COL.budgetWarn : COL.hitGreen;
    const roleTxt = new P.Text({
      text: role + " · sensitivity " + SENSITIVITY[role],
      style: { fontFamily: "Helvetica Neue, Arial", fontSize: 8.5, fill: sensitivityColor }
    });
    roleTxt.position.set(8, 21);
    row.addChild(roleTxt);

    /* Precision label (right side) */
    const precTxt = new P.Text({
      text: PRECISIONS[0].name,
      style: { fontFamily: "Helvetica Neue, Arial", fontSize: 12, fontWeight: "700", fill: PRECISIONS[0].stroke }
    });
    precTxt.anchor.set(1, 0);
    precTxt.position.set(DIAL_W - 8, 4);
    row.addChild(precTxt);

    /* Stop indicator dots underneath the precision label */
    const dotsContainer = new P.Container();
    dotsContainer.position.set(DIAL_W - 8, 24);
    row.addChild(dotsContainer);
    const stopDots = [];
    for (let k = 0; k < PRECISIONS.length; k++) {
      const dot = new P.Graphics();
      dot.position.set(-k * 9, 0);
      dotsContainer.addChild(dot);
      stopDots.unshift(dot);    // unshift so stopDots[0] = leftmost = fp32
    }

    /* Make whole row interactive */
    row.eventMode = "static";
    row.cursor = "pointer";
    row.hitArea = new P.Rectangle(0, 0, DIAL_W, DIAL_H - 4);
    row.on("pointerdown", () => cycleLayer(i));
    row.on("pointerover", () => { row.scale.set(1.02); });
    row.on("pointerout",  () => { row.scale.set(1); });

    panelLayer.addChild(row);
    dialRows.push({ row, pill, nameTxt, precTxt, roleTxt, stopDots });
  }

  /* --- Budget bar --- */
  const BUDGET_Y = DIALS_Y + NUM_LAYERS * DIAL_H + 6;
  const budgetLabel = new P.Text({
    text: "budget",
    style: { fontFamily: "Helvetica Neue, Arial", fontSize: 10, fontWeight: "700", fill: COL.text }
  });
  budgetLabel.position.set(DIAL_X, BUDGET_Y);
  panelLayer.addChild(budgetLabel);

  const budgetVal = new P.Text({
    text: "192 / 96 bits",
    style: { fontFamily: "Helvetica Neue, Arial", fontSize: 10, fill: COL.muted }
  });
  budgetVal.anchor.set(1, 0);
  budgetVal.position.set(DIAL_X + DIAL_W, BUDGET_Y);
  panelLayer.addChild(budgetVal);

  const budgetBarBg = new P.Graphics();
  budgetBarBg.roundRect(DIAL_X, BUDGET_Y + 16, DIAL_W, 8, 4).fill({ color: COL.faint });
  panelLayer.addChild(budgetBarBg);
  const budgetBar = new P.Graphics();
  panelLayer.addChild(budgetBar);

  /* --- Accuracy meter --- */
  const ACC_Y = BUDGET_Y + 36;
  const accLabel = new P.Text({
    text: "estimated accuracy",
    style: { fontFamily: "Helvetica Neue, Arial", fontSize: 10, fontWeight: "700", fill: COL.text }
  });
  accLabel.position.set(DIAL_X, ACC_Y);
  panelLayer.addChild(accLabel);

  const accVal = new P.Text({
    text: "100%",
    style: { fontFamily: "Helvetica Neue, Arial", fontSize: 10, fontWeight: "700", fill: COL.hitGreen }
  });
  accVal.anchor.set(1, 0);
  accVal.position.set(DIAL_X + DIAL_W, ACC_Y);
  panelLayer.addChild(accVal);

  const accBarBg = new P.Graphics();
  accBarBg.roundRect(DIAL_X, ACC_Y + 16, DIAL_W, 8, 4).fill({ color: COL.faint });
  panelLayer.addChild(accBarBg);
  const accBar = new P.Graphics();
  panelLayer.addChild(accBar);

  /* --- Fire button --- */
  const FIRE_Y = H - 70;
  const fireBtn = new P.Container();
  fireBtn.position.set(DIAL_X, FIRE_Y);
  fireBtn.eventMode = "static";
  fireBtn.cursor = "pointer";
  const fireBtnBg = new P.Graphics();
  const fireBtnTxt = new P.Text({
    text: "FIRE",
    style: { fontFamily: "Helvetica Neue, Arial", fontSize: 16, fontWeight: "800", fill: COL.white }
  });
  fireBtnTxt.anchor.set(0.5, 0.5);
  fireBtnTxt.position.set(DIAL_W / 2, 18);
  const fireBtnSub = new P.Text({
    text: "(space or click target)",
    style: { fontFamily: "Helvetica Neue, Arial", fontSize: 9, fill: 0xffe9ed }
  });
  fireBtnSub.anchor.set(0.5, 0.5);
  fireBtnSub.position.set(DIAL_W / 2, 34);
  fireBtn.addChild(fireBtnBg, fireBtnTxt, fireBtnSub);
  fireBtn.hitArea = new P.Rectangle(0, 0, DIAL_W, 44);
  fireBtn.on("pointerdown", () => fireShot());
  panelLayer.addChild(fireBtn);

  /* --- Target zone (right side) --- */
  const targetBg = new P.Graphics();
  targetBg.roundRect(TARGET.x, TARGET.y, TARGET.w, TARGET.h, 6)
          .fill({ color: COL.white })
          .stroke({ color: COL.panelStroke, width: 1 });
  /* Faint downrange grid + horizon line so the zone reads as "downrange",
     not "empty white box". Drawn once at boot, never updated. */
  const gridStep = 40;
  for (let gx = TARGET.x + gridStep; gx < TARGET.x + TARGET.w; gx += gridStep) {
    targetBg.moveTo(gx, TARGET.y).lineTo(gx, TARGET.y + TARGET.h)
            .stroke({ color: 0xeef1f4, width: 1, alpha: 0.6 });
  }
  for (let gy = TARGET.y + gridStep; gy < TARGET.y + TARGET.h; gy += gridStep) {
    targetBg.moveTo(TARGET.x, gy).lineTo(TARGET.x + TARGET.w, gy)
            .stroke({ color: 0xeef1f4, width: 1, alpha: 0.6 });
  }
  /* Horizon strip across vertical midline */
  targetBg.moveTo(TARGET.x, TARGET.y + TARGET.h / 2)
          .lineTo(TARGET.x + TARGET.w, TARGET.y + TARGET.h / 2)
          .stroke({ color: 0xd6dde3, width: 1, alpha: 0.7 });
  bgLayer.addChild(targetBg);

  /* The bullseye lives inside targetWrap so we can filter it */
  const ringOuter = new P.Graphics();
  const ringMid   = new P.Graphics();
  const ringInner = new P.Graphics();
  const bullseye  = new P.Graphics();
  const ringRotator = new P.Graphics();   // decorative outer rotating ring
  targetWrap.addChild(ringRotator, ringOuter, ringMid, ringInner, bullseye);

  function drawBullseye(cx, cy) {
    ringOuter.clear()
      .circle(cx, cy, TARGET_RADIUS)
      .fill({ color: COL.ringOuter })
      .stroke({ color: 0xc8c0a8, width: 1, alpha: 0.6 });
    ringMid.clear()
      .circle(cx, cy, MID_RADIUS)
      .fill({ color: COL.ringMid })
      .stroke({ color: 0xa8b8c8, width: 1, alpha: 0.6 });
    ringInner.clear()
      .circle(cx, cy, BULLSEYE_RADIUS + 4)
      .fill({ color: COL.ringInner });
    bullseye.clear()
      .circle(cx, cy, BULLSEYE_RADIUS)
      .fill({ color: COL.bullseye });
    /* Decorative rotating outer ring (just visual flavour) */
    ringRotator.clear();
    for (let k = 0; k < 8; k++) {
      const ang = (k / 8) * Math.PI * 2 + state.targetPhase;
      const rx = cx + Math.cos(ang) * (TARGET_RADIUS + 6);
      const ry = cy + Math.sin(ang) * (TARGET_RADIUS + 6);
      ringRotator.circle(rx, ry, 1.5).fill({ color: 0xc8d3dc, alpha: 0.6 });
    }
  }

  /* --- True-target reveal (after a miss) --- */
  const trueTargetReveal = new P.Graphics();
  const trueTargetLabel = new P.Text({
    text: "true target",
    style: { fontFamily: "Helvetica Neue, Arial", fontSize: 9.5, fontStyle: "italic", fill: COL.missRed }
  });
  trueTargetLabel.anchor.set(0.5, 1);
  trueTargetLabel.alpha = 0;
  overlayLayer.addChild(trueTargetReveal, trueTargetLabel);

  /* --- Shot trail line --- */
  const shotTrail = new P.Graphics();
  overlayLayer.addChild(shotTrail);

  /* --- Crosshair --- */
  const crosshair = new P.Graphics();
  overlayLayer.addChild(crosshair);

  function drawCrosshair() {
    const x = state.crosshairX, y = state.crosshairY;
    const overBudget = bitsUsed() > MAX_BUDGET;
    const goodConfig = !overBudget && state.accuracy >= 80;
    const colour = overBudget ? COL.missRed : (goodConfig ? COL.hitGreen : COL.fireGood);
    crosshair.clear()
      .moveTo(x - 13, y).lineTo(x - 5, y)
      .moveTo(x + 5, y).lineTo(x + 13, y)
      .moveTo(x, y - 13).lineTo(x, y - 5)
      .moveTo(x, y + 5).lineTo(x, y + 13)
      .stroke({ color: colour, width: 1.6 })
      .circle(x, y, 3.5).stroke({ color: colour, width: 1.6 });
  }

  /* --- Tutorial banner (bottom of canvas, fades out after first cycle) --- */
  const tutText = new P.Text({
    text: "click any layer to cycle precision · move mouse to aim · SPACE to fire",
    style: { fontFamily: "Helvetica Neue, Arial", fontSize: 11, fontWeight: "600", fill: COL.fireGood }
  });
  tutText.anchor.set(0.5, 0.5);
  tutText.position.set(TARGET.cx, H - 26);
  hudLayer.addChild(tutText);

  /* --- Game-over overlay (built on demand) --- */
  const gameOverOverlay = new P.Container();
  gameOverOverlay.visible = false;
  hudLayer.addChild(gameOverOverlay);

  /* ============================================================
     Logic
     ============================================================ */

  function bitsUsed() {
    let s = 0;
    for (let i = 0; i < NUM_LAYERS; i++) s += PRECISIONS[state.layers[i].precisionIdx].bits;
    return s;
  }

  /* Recompute blur / jitter / drift / accuracy from current layer config.
     Accuracy model is intentionally simple but qualitatively correct:
     - Each layer contributes a degradation = bits-reduction × role-weight.
     - Edge at int4 dominates (the cliff).
     - 100% perfect, drops to ~40% at all-int4-edges + int4-everywhere.    */
  function recomputeEffects() {
    state.blur = 0;
    state.jitter = 0;
    state.driftX = 0;
    state.driftY = 0;
    let degradation = 0;

    for (let i = 0; i < NUM_LAYERS; i++) {
      const p = state.layers[i].precisionIdx;
      const bitsRed = (32 - PRECISIONS[p].bits) / 4;   // 0, 4, 6, 7
      if (bitsRed === 0) continue;
      const role = LAYER_ROLES[i];
      if (role === "edge") {
        const sign = (i === 0) ? 1 : -1;
        if (p >= 3) {
          state.driftX += sign * 65;
          state.driftY += sign * 40;
          degradation += 22;       // the cliff
        } else if (p >= 2) {
          state.driftX += sign * 14;
          state.driftY += sign * 8;
          degradation += 6;
        } else {
          degradation += 2;
        }
      } else if (role === "attn") {
        state.jitter += bitsRed * 1.6;
        degradation += bitsRed * 1.2;
      } else if (role === "ffn") {
        state.blur += bitsRed * 0.95;
        degradation += bitsRed * 0.6;
      }
    }
    state.accuracy = Math.max(20, Math.min(100, 100 - degradation));
    /* Apply / remove BlurFilter on the target wrap.
       BlurFilter ships with core PIXI (not pixi-filters), so it's
       available immediately without waiting for the lazy-loaded
       filters bundle. */
    if (state.blur > 0.4) {
      const radius = Math.min(8, state.blur);
      if (!targetWrap._blurFilter) {
        targetWrap._blurFilter = new PIXI.BlurFilter({ strength: radius, quality: 4 });
        targetWrap.filters = [targetWrap._blurFilter];
      } else {
        targetWrap._blurFilter.strength = radius;
      }
    } else if (targetWrap._blurFilter) {
      targetWrap.filters = [];
      targetWrap._blurFilter = null;
    }
  }

  function cycleLayer(idx) {
    if (state.over) return;
    const layer = state.layers[idx];
    layer.precisionIdx = (layer.precisionIdx + 1) % PRECISIONS.length;
    recomputeEffects();
    /* Animate the dial-cell pill */
    const row = dialRows[idx].row;
    tween(row, "scale.x", 1.06, 1, 220, "outElastic");
    tween(row, "scale.y", 1.06, 1, 220, "outElastic");
    pop(overlayLayer, DIAL_X + DIAL_W / 2 - 12, DIALS_Y + idx * DIAL_H + 22, PRECISIONS[layer.precisionIdx].stroke, { r: 18, ms: 320 });
    if (state.showTutorial) state.showTutorial = false;
  }

  function fireShot() {
    if (state.over || state.shotsLeft <= 0) return;
    if (bitsUsed() > MAX_BUDGET) {
      floatText(overlayLayer, state.crosshairX, state.crosshairY - 24, "over budget — reduce bits", COL.missRed, { size: 11, lifeMs: 1200 });
      shake(overlayLayer, 5, 200);
      flash(stage, COL.missRed, 140, 0.18);
      return;
    }
    state.shotsLeft--;
    const dx = state.trueTargetX - state.crosshairX;
    const dy = state.trueTargetY - state.crosshairY;
    const dist = Math.hypot(dx, dy);
    /* Score by zone */
    let zone = 0;     // miss
    if (dist < BULLSEYE_RADIUS) zone = 3;
    else if (dist < MID_RADIUS) zone = 2;
    else if (dist < TARGET_RADIUS) zone = 1;
    state.score += zone;
    state.perShotZones.push(zone);

    state.lastShot = {
      x: state.crosshairX,
      y: state.crosshairY,
      tx: state.trueTargetX,
      ty: state.trueTargetY,
      zone, ms: 600
    };
    state.shotFlashMs = 400;
    if (zone === 0) state.revealMs = 900;       // show true target on miss

    const fb = zone === 3 ? "✧ bullseye +3" :
               zone === 2 ? "✓ inner +2" :
               zone === 1 ? "✓ outer +1" : "✗ miss";
    const fbColor = zone === 0 ? COL.missRed :
                    zone === 3 ? COL.bullseye : COL.hitGreen;
    floatText(overlayLayer, state.crosshairX, state.crosshairY - 16, fb, fbColor, { size: 13, lifeMs: 1100 });

    if (zone > 0) {
      const burstCount = zone === 3 ? 26 : zone === 2 ? 16 : 10;
      burst(overlayLayer, state.crosshairX, state.crosshairY, fbColor, burstCount, { speed: 2.6, lifeMs: 750 });
      pop(overlayLayer, state.crosshairX, state.crosshairY, fbColor, { r: 14 + zone * 6, ms: 380 });
    } else {
      burst(overlayLayer, state.crosshairX, state.crosshairY, COL.missRed, 8);
      pop(overlayLayer, state.crosshairX, state.crosshairY, COL.missRed, { r: 18 });
      shake(overlayLayer, 3, 180);
    }
    /* Move true target for the next shot */
    movetrueTarget();

    if (state.shotsLeft <= 0) {
      state.over = true;
      state.won = state.score >= SHIP_THRESHOLD;
      flash(stage, state.won ? COL.hitGreen : COL.missRed, 360, 0.25);
      endGame();
    }
  }

  function endGame() {
    if (state.score > alltimeBestRef.v) {
      alltimeBestRef.v = state.score;
      bestScore.set("quantization", state.score);
    }
    drawGameOver();
    if (opts.onGameOver) {
      opts.onGameOver({
        score:        state.score,
        won:          state.won,
        shots:        SHOTS_PER_ROUND,
        bitsUsed:     bitsUsed(),
        accuracy:     state.accuracy,
        precisions:   state.layers.map(l => l.precisionIdx),
        zones:        state.perShotZones,
        emojiGrid:    buildEmojiGrid(),
        alltimeBest:  alltimeBestRef.v
      });
    }
  }

  function buildEmojiGrid() {
    let row1 = "";
    for (const z of state.perShotZones) {
      row1 += z === 3 ? "🎯" : z === 2 ? "🟢" : z === 1 ? "🟡" : "⚫";
    }
    let row2 = "";
    for (let i = 0; i < NUM_LAYERS; i++) {
      const p = state.layers[i].precisionIdx;
      row2 += p === 0 ? "🟦" : p === 1 ? "🟩" : p === 2 ? "🟧" : "🟥";
    }
    return row1 + "\n" + row2 + "  ← per-layer precision";
  }

  /* ============================================================
     Input
     ============================================================ */

  /* Mouse aim — only respond inside target box */
  canvas.addEventListener("pointermove", (e) => {
    if (state.over) return;
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (canvas.width / rect.width);
    const py = (e.clientY - rect.top) * (canvas.height / rect.height);
    if (px >= TARGET.x && px <= TARGET.x + TARGET.w &&
        py >= TARGET.y && py <= TARGET.y + TARGET.h) {
      state.crosshairX = px;
      state.crosshairY = py;
    }
  });

  /* Click on target zone = aim + fire (so casual players don't need keyboard) */
  canvas.addEventListener("pointerdown", (e) => {
    if (state.over) {
      if (opts.onRetry) opts.onRetry();
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (canvas.width / rect.width);
    const py = (e.clientY - rect.top) * (canvas.height / rect.height);
    /* If click is inside target zone (and not on the panel), fire */
    if (px >= TARGET.x && px <= TARGET.x + TARGET.w &&
        py >= TARGET.y && py <= TARGET.y + TARGET.h) {
      state.crosshairX = px;
      state.crosshairY = py;
      fireShot();
    }
  });

  function handleKeydown(e) {
    if (e.key === "r" || e.key === "R") {
      e.preventDefault();
      if (opts.onRetry) opts.onRetry();
      return;
    }
    if (state.over) return;
    if (e.key === " " || e.key === "Enter") { e.preventDefault(); fireShot(); }
    if (e.key >= "1" && e.key <= "6") { e.preventDefault(); cycleLayer(parseInt(e.key, 10) - 1); }
    /* Optional arrow-key aim for accessibility */
    const step = 8;
    if (e.key === "ArrowLeft")  { e.preventDefault(); state.crosshairX = Math.max(TARGET.x, state.crosshairX - step); }
    if (e.key === "ArrowRight") { e.preventDefault(); state.crosshairX = Math.min(TARGET.x + TARGET.w, state.crosshairX + step); }
    if (e.key === "ArrowUp")    { e.preventDefault(); state.crosshairY = Math.max(TARGET.y, state.crosshairY - step); }
    if (e.key === "ArrowDown")  { e.preventDefault(); state.crosshairY = Math.min(TARGET.y + TARGET.h, state.crosshairY + step); }
  }
  window.addEventListener("keydown", handleKeydown);

  /* ============================================================
     Frame loop
     ============================================================ */

  recomputeEffects();

  app.ticker.add((ticker) => {
    const dt = ticker.deltaMS;

    state.targetPhase += dt * 0.0008;

    /* Tween each dial's animatedIdx toward its target */
    for (let i = 0; i < NUM_LAYERS; i++) {
      const layer = state.layers[i];
      const k = 0.18;
      layer.animatedIdx += (layer.precisionIdx - layer.animatedIdx) * k * (dt / 16);
    }

    /* Decay shot/reveal flashes */
    state.shotFlashMs = Math.max(0, state.shotFlashMs - dt);
    state.revealMs    = Math.max(0, state.revealMs - dt);

    /* Re-draw the bullseye (jitter applied in displayed position) */
    const jitterX = state.jitter > 0
      ? (Math.sin(state.targetPhase * 9) + (rand() - 0.5) * 0.6) * state.jitter
      : 0;
    const jitterY = state.jitter > 0
      ? (Math.cos(state.targetPhase * 7) + (rand() - 0.5) * 0.6) * state.jitter
      : 0;
    const displayX = state.trueTargetX + state.driftX + jitterX;
    const displayY = state.trueTargetY + state.driftY + jitterY;
    drawBullseye(displayX, displayY);

    /* Crosshair pulse over budget */
    drawCrosshair();

    /* True-target reveal on miss */
    if (state.revealMs > 0) {
      const a = state.revealMs / 900;
      trueTargetReveal.clear()
        .circle(state.lastShot.tx, state.lastShot.ty, TARGET_RADIUS)
        .stroke({ color: COL.missRed, width: 2, alpha: a * 0.85 });
      trueTargetLabel.position.set(state.lastShot.tx, state.lastShot.ty - TARGET_RADIUS - 6);
      trueTargetLabel.alpha = a;
    } else {
      trueTargetReveal.clear();
      trueTargetLabel.alpha = 0;
    }

    /* Shot trail */
    if (state.lastShot && state.shotFlashMs > 0) {
      const a = state.shotFlashMs / 400;
      shotTrail.clear()
        .moveTo(DIAL_X + DIAL_W / 2, FIRE_Y + 22)   // start at fire button
        .lineTo(state.lastShot.x, state.lastShot.y)
        .stroke({ color: state.lastShot.zone > 0 ? COL.hitGreen : COL.missRed, width: 1.5, alpha: a * 0.6 });
    } else {
      shotTrail.clear();
    }

    /* Update HUD strip */
    updateDials();
    updateBudget();
    updateAccuracy();
    updateFireButton();

    /* Tutorial fade */
    if (!state.showTutorial && tutText.alpha > 0) {
      tutText.alpha = Math.max(0, tutText.alpha - dt / 600);
    }

    /* Score callback */
    if (opts.onScoreChange && !state.over) {
      opts.onScoreChange({
        bitsUsed:    bitsUsed(),
        budget:      MAX_BUDGET,
        shotsLeft:   state.shotsLeft,
        score:       state.score,
        accuracy:    Math.round(state.accuracy),
        alltimeBest: alltimeBestRef.v
      });
    }
  });

  /* ============================================================
     HUD/dial draw helpers (called every frame)
     ============================================================ */

  function updateDials() {
    for (let i = 0; i < NUM_LAYERS; i++) {
      const r = dialRows[i];
      const p = state.layers[i].precisionIdx;
      const animP = state.layers[i].animatedIdx;
      const prec = PRECISIONS[p];
      /* Pill colour reflects current precision */
      r.pill.clear()
        .roundRect(0, 0, DIAL_W, DIAL_H - 4, 6)
        .fill({ color: prec.color })
        .stroke({ color: prec.stroke, width: 1.5 });
      r.precTxt.text = prec.name;
      r.precTxt.style.fill = prec.stroke;
      /* Stop dots — current stop is filled, others are outlined */
      for (let k = 0; k < PRECISIONS.length; k++) {
        const dot = r.stopDots[k];
        const active = (k === p);
        const closeness = 1 - Math.min(1, Math.abs(k - animP));
        dot.clear()
          .circle(0, 0, 3)
          .fill({ color: active ? PRECISIONS[k].stroke : 0xc8d3dc, alpha: active ? 1 : 0.4 + 0.5 * closeness });
      }
    }
  }

  function updateBudget() {
    const used = bitsUsed();
    const frac = Math.min(1, used / MAX_BUDGET);
    const over = used > MAX_BUDGET;
    const colour = over ? COL.budgetOver : (frac > 0.9 ? COL.budgetWarn : COL.budgetGood);
    budgetBar.clear()
      .roundRect(DIAL_X, BUDGET_Y + 16, DIAL_W * frac, 8, 4)
      .fill({ color: colour });
    budgetVal.text = used + " / " + MAX_BUDGET + " bits" + (over ? " — over!" : "");
    budgetVal.style.fill = over ? COL.budgetOver : COL.muted;
  }

  function updateAccuracy() {
    const a = state.accuracy / 100;
    const colour = a >= 0.85 ? COL.hitGreen : a >= 0.65 ? COL.budgetWarn : COL.budgetOver;
    accBar.clear()
      .roundRect(DIAL_X, ACC_Y + 16, DIAL_W * a, 8, 4)
      .fill({ color: colour });
    accVal.text = Math.round(state.accuracy) + "%";
    accVal.style.fill = colour;
  }

  function updateFireButton() {
    const used = bitsUsed();
    const canFire = state.shotsLeft > 0 && !state.over && used <= MAX_BUDGET;
    fireBtnBg.clear()
      .roundRect(0, 0, DIAL_W, 44, 7)
      .fill({ color: canFire ? COL.fireGood : COL.fireBad });
    fireBtnTxt.text = state.shotsLeft > 0 ? "FIRE  ·  " + state.shotsLeft + " left" : "OUT OF SHOTS";
    fireBtnTxt.style.fill = canFire ? COL.white : 0xefefef;
  }

  /* ============================================================
     Game-over screen
     ============================================================ */

  function drawGameOver() {
    gameOverOverlay.visible = true;
    const bg = new P.Graphics();
    bg.rect(0, 0, W, H).fill({ color: COL.white, alpha: 0.93 });
    const titleStr = state.won ? "🏆 model shipped!" : "accuracy below spec";
    const titleColor = state.won ? COL.hitGreen : COL.fireGood;
    const t1 = new P.Text({
      text: titleStr,
      style: { fontFamily: "Helvetica Neue, Arial", fontSize: 26, fontWeight: "800", fill: titleColor }
    });
    t1.anchor.set(0.5, 0.5); t1.position.set(W / 2, H / 2 - 30);
    if (filters && filters.GlowFilter && state.won) {
      t1.filters = [new filters.GlowFilter({ distance: 16, outerStrength: 2, innerStrength: 0.4, color: COL.hitGreen, quality: 0.4 })];
    }
    const t2 = new P.Text({
      text: "score " + state.score + " / " + (SHOTS_PER_ROUND * 3) +
            " · " + bitsUsed() + " bits · " + Math.round(state.accuracy) + "% acc",
      style: { fontFamily: "Helvetica Neue, Arial", fontSize: 14, fill: COL.text }
    });
    t2.anchor.set(0.5, 0.5); t2.position.set(W / 2, H / 2);
    const t3 = new P.Text({
      text: "best " + alltimeBestRef.v + " · tap or press R to retry",
      style: { fontFamily: "Helvetica Neue, Arial", fontSize: 11, fontStyle: "italic", fill: COL.muted }
    });
    t3.anchor.set(0.5, 0.5); t3.position.set(W / 2, H / 2 + 28);
    gameOverOverlay.addChild(bg, t1, t2, t3);
  }

  /* ============================================================
     Aha card / share API
     ============================================================ */

  return {
    id: "quantization",
    name: "Quantization Sharp Shot",
    ahaLabel: "You just played at",
    ahaText:
      "Quantization error isn't uniform. Edge layers (embedding & output) at int4 cause systematic bias " +
      "(the target drifted away from your sight); attention layers amplify noise (jitter); FFN layers " +
      "mostly tolerate it (just blur). Real quantization navigates exactly this asymmetry — mixed-precision " +
      "allocation (HAWQ, Dong et al. 2019) and calibration-based rounding (GPTQ, AWQ) exist to manage it.",
    ahaLink: { href: "https://arxiv.org/abs/2208.07339", label: "LLM.int8 — the edge-layer cliff →" },
    buildShareText(r) {
      let layerEmoji = "";
      const precs = r.precisions || state.layers.map(l => l.precisionIdx);
      for (const p of precs) {
        layerEmoji += p === 0 ? "🟦" : p === 1 ? "🟩" : p === 2 ? "🟧" : "🟥";
      }
      return "MLSysBook Playground · Quantization Sharp Shot · Day " + dayNumber() + "\n" +
        (r.won ? "🏆 shipped" : "✗ off-spec") +
        " · " + r.score + "/" + (SHOTS_PER_ROUND * 3) + " pts · " +
        r.bitsUsed + " bits · " + Math.round(r.accuracy) + "% acc\n" +
        r.emojiGrid + "\n" +
        "play → mlsysbook.ai/games/quantization/";
    },
    destroy() {
      window.removeEventListener("keydown", handleKeydown);
      app.destroy(true, { children: true, texture: true });
    }
  };
}
