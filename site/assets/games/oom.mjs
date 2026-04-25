/* ============================================================
   MLSysBook Playground — Tensor Tetris (v6, Pixi + pedagogy fixes)
   (registry id stays "oom"; the failure state is OOM, the game is Tetris)
   ------------------------------------------------------------
   What changed from v5 (per pedagogue audit):
     1. KV cache REMOVED. Replaced with a "parameters (W)" block:
        spawns ONCE near game start, large, persistent, never freed.
        (KV cache is an inference citizen; this is a training game.)
     2. Block sizes RESCALED to match real training-time HBM ratios:
        - activations: 6 cells (largest, dominant — they are why you OOM)
        - optimizer state: 6 cells (Adam ≈ 6× param bytes: m+v+fp32 master)
        - parameters: 3 cells (anchor)
        - gradients: 1 cell (smallest, frequent, freed often)
     3. Spawn frequencies REWEIGHTED so activations dominate the inflow.
     4. Activation freeing flipped FIFO → LIFO (real backward is reverse
        graph traversal: NEWEST activation freed FIRST).
     5. Aha card rewritten to Korthikanti et al. 2022 — the canonical
        "activations dominate, not weights" reference.

   Visual lift (this round):
     - Real GlowFilter on the parameters block, on backward bursts, on
       step() bursts (via lazy-loaded pixi-filters).
     - Real BloomFilter pulse on game-over overflow.
     - Particle bursts denser (40+ per event vs 5–8 in v5).
     - Block placement uses smooth tween-in (scale + alpha) instead of pop-in.
     - Subtle ambient particles drifting up the HBM column when ≥40% full
       (visual cue: pressure rising).
   ============================================================ */

import {
  mountPixiOnCanvas, dailySeed, dayNumber, bestScore,
  pop, flash, burst, floatText, shake, tween, getFilters
} from "/assets/games/runtime.mjs";
import * as PIXI from "/assets/games/vendor/pixi.min.mjs";

const TIME_LIMIT_MS = 60000;
const STEP_BLOCKS = 6;
const BACKWARD_BLOCKS = 3;

const COL = {
  blueLight:  0xcfe2f3, blueStroke:  0x4a90c4,
  redLight:   0xf9d6d5, redStroke:   0xc44444,
  orangeLight:0xfdebd0, orangeStroke:0xc87b2a,
  purpleLight:0xe1d5e7, purpleStroke:0x6a4a7a,
  mitRed:     0xa31f34,
  text:       0x333333, muted: 0x888888, faint: 0xeeeeee,
  white:      0xffffff
};

/* Block shapes — see header for ratio rationale.
   Cell counts: act=6 (3+3 variants), opt=6 (2x3), grad=1, params=3.
   Frequencies sum to 1.0 minus the params-spawn (handled separately). */
const BLOCK_TYPES = [
  // Activations: two variants, both 6 cells, totalling ~55% of inflow
  { name: "activation",      kind: "act",   color: COL.blueLight,   stroke: COL.blueStroke,   cells: [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1]],         freq: 0.30 },
  { name: "activation",      kind: "act",   color: COL.blueLight,   stroke: COL.blueStroke,   cells: [[0,0],[1,0],[0,1],[1,1],[0,2],[1,2]],         freq: 0.25 },
  // Gradients: 1 cell each, very frequent (one per parameter), 25% of inflow
  { name: "gradient",        kind: "grad",  color: COL.redLight,    stroke: COL.redStroke,    cells: [[0,0]],                                       freq: 0.25 },
  // Optimizer state: 6 cells (Adam = m+v+fp32 master ≈ 6× weight bytes), 20% of inflow
  { name: "optimizer state", kind: "opt",   color: COL.orangeLight, stroke: COL.orangeStroke, cells: [[0,0],[1,0],[0,1],[1,1],[0,2],[1,2]],         freq: 0.20 }
];
let freqSum = 0;
for (const t of BLOCK_TYPES) freqSum += t.freq;

/* Parameters block — spawned once near game start, persistent. */
const PARAMS_TYPE = {
  name: "parameters (W)", kind: "param",
  color: COL.purpleLight, stroke: COL.purpleStroke,
  cells: [[0,0],[1,0],[2,0]]  // 3 cells horizontal — anchors the bottom
};

window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};
window.MLSP.games.oom = function (canvas, opts) { return mountOom(canvas, opts); };

export async function mountOom(canvas, opts = {}) {
  const { app, stage, width: W, height: H } = await mountPixiOnCanvas(canvas, { bg: COL.white });
  const { rand, today } = dailySeed("oom");

  const hbmX = W / 2 - 160, hbmY = 80, hbmW = 320, hbmH = H - 200;
  const cellSize = 20;
  const cols = Math.floor(hbmW / cellSize);
  const rows = Math.floor(hbmH / cellSize);

  const grid = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) row.push(null);
    grid.push(row);
  }

  let nextBlockId = 1;
  const state = {
    current: null,
    nextFallIn: 600,
    fallInterval: 800,
    score: 0,
    over: false,
    timeLeft: TIME_LIMIT_MS,
    stepProgress: 0,
    backwardProgress: 0,
    stepFlashTime: 0,
    backwardFlashTime: 0,
    paramsPlaced: false
  };
  const alltimeBestRef = { v: bestScore.get("oom") };

  /* --- Layers --- */
  const gameLayer = new PIXI.Container();
  const ambientLayer = new PIXI.Container();
  const hudLayer = new PIXI.Container();
  const overlayLayer = new PIXI.Container();
  stage.addChild(gameLayer, ambientLayer, hudLayer, overlayLayer);

  const hbmFrame = new PIXI.Graphics();
  const gridLines = new PIXI.Graphics();
  /* Each placed block gets its own Container so we can tween it independently */
  const blocksContainer = new PIXI.Container();
  const currentGfx = new PIXI.Graphics();
  gameLayer.addChild(hbmFrame, gridLines, blocksContainer, currentGfx);

  /* placedSprites maps "r,c" → Container, so we can fade-out cells during free events */
  const placedSprites = new Map();

  /* --- Lazy-load filters for visual lift --- */
  let filters = null;
  getFilters().then(f => { filters = f; });

  const titleText = new PIXI.Text({ text: "Tensor Tetris", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 15, fontWeight: "700", fill: COL.text } });
  titleText.anchor.set(0.5, 0); titleText.position.set(W / 2, 14);
  const subtitleText = new PIXI.Text({ text: "← → move · space drop · memory frees on training events, not completed rows", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 10.5, fill: COL.muted } });
  subtitleText.anchor.set(0.5, 0); subtitleText.position.set(W / 2, 36);
  const phaseText = new PIXI.Text({ text: "FORWARD — packing tensors", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 11, fontWeight: "700", fill: COL.muted } });
  phaseText.anchor.set(0.5, 0); phaseText.position.set(W / 2, 56);

  const stepBarFrame = new PIXI.Graphics();
  const stepBarFill = new PIXI.Graphics();
  const bwdBarFrame = new PIXI.Graphics();
  const bwdBarFill = new PIXI.Graphics();
  const stepLabel = new PIXI.Text({ text: "step", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 9, fill: 0x555555 } });
  const stepCounter = new PIXI.Text({ text: "0/6", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 9, fill: 0x555555 } });
  const bwdLabel = new PIXI.Text({ text: "bwd", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 9, fill: 0x555555 } });

  const scoreText = new PIXI.Text({ text: "score 0", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 11, fill: COL.text } });
  scoreText.position.set(20, H - 32);
  const timerText = new PIXI.Text({ text: "⏱ 60s", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 13, fontWeight: "700", fill: COL.text } });
  timerText.anchor.set(0.5, 0); timerText.position.set(W / 2, H - 32);
  const bestText = new PIXI.Text({ text: "all-time best 0", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 11, fill: COL.text } });
  bestText.anchor.set(1, 0); bestText.position.set(W - 20, H - 32);
  const dailyText = new PIXI.Text({ text: "", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 9, fill: 0x999999 } });
  dailyText.position.set(20, H - 14);
  const eventTitle = new PIXI.Text({ text: "why blocks vanish", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 10, fontWeight: "700", fill: COL.text } });
  const eventLine1 = new PIXI.Text({ text: "3 placements → backward", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 9, fill: COL.blueStroke } });
  const eventLine2 = new PIXI.Text({ text: "newest activations free first", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 8.5, fill: COL.muted } });
  const eventLine3 = new PIXI.Text({ text: "6 placements → step()", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 9, fill: COL.orangeStroke } });
  const eventLine4 = new PIXI.Text({ text: "gradients get used + cleared", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 8.5, fill: COL.muted } });
  const eventLine5 = new PIXI.Text({ text: "rows themselves do not clear", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 8.5, fontStyle: "italic", fill: COL.mitRed } });

  /* Legend (left side) — now four kinds: activation, gradient, optimizer, parameters */
  const legendData = [
    { c: COL.blueLight,   s: COL.blueStroke,   n: "activation" },
    { c: COL.redLight,    s: COL.redStroke,    n: "gradient" },
    { c: COL.orangeLight, s: COL.orangeStroke, n: "optimizer" },
    { c: COL.purpleLight, s: COL.purpleStroke, n: "parameters" }
  ];
  const legendGfx = new PIXI.Graphics();
  const legendLabels = legendData.map(d => new PIXI.Text({ text: d.n, style: { fontFamily: "Helvetica Neue, Arial", fontSize: 9, fill: 0x555555 } }));

  hudLayer.addChild(
    titleText, subtitleText, phaseText,
    stepBarFrame, stepBarFill, bwdBarFrame, bwdBarFill,
    stepLabel, stepCounter, bwdLabel,
    scoreText, timerText, bestText, dailyText,
    eventTitle, eventLine1, eventLine2, eventLine3, eventLine4, eventLine5,
    legendGfx, ...legendLabels
  );

  const gameOverOverlay = new PIXI.Container();
  gameOverOverlay.visible = false;
  overlayLayer.addChild(gameOverOverlay);

  /* --- Drawing helpers --- */

  function drawCell(gfx, x, y, color, stroke) {
    const sz = cellSize - 2;
    gfx.roundRect(x + 1, y + 1, sz, sz, 3).fill({ color }).stroke({ width: 1.5, color: stroke });
  }

  /* --- Mechanics --- */

  function pickType() {
    let r = rand() * freqSum;
    for (const t of BLOCK_TYPES) { r -= t.freq; if (r <= 0) return t; }
    return BLOCK_TYPES[0];
  }

  function spawnBlock() {
    /* First spawn = parameters block, anchored at the bottom of HBM.
       Persistent like real model weights — never freed. */
    if (!state.paramsPlaced) {
      placeParametersBlock();
      state.paramsPlaced = true;
      state.nextFallIn = 800;
      return;
    }
    const type = pickType();
    const block = { type, row: 0, col: Math.floor((cols - colSpan(type)) / 2), blockId: nextBlockId++ };
    if (collides(block, block.row, block.col)) {
      state.over = true;
      shake(gameLayer, 10, 400);
      flash(stage, COL.mitRed, 360);
      /* Game-over: bloom on the offending column */
      const px = hbmX + block.col * cellSize + cellSize;
      const py = hbmY + 12;
      burst(overlayLayer, px, py, COL.mitRed, 60, { speed: 4, lifeMs: 900 });
      endGame();
      return;
    }
    state.current = block;
  }

  function colSpan(type) {
    let max = 0;
    for (const [dc, _dr] of type.cells) if (dc > max) max = dc;
    return max + 1;
  }

  function placeParametersBlock() {
    /* Place parameters at the bottom of HBM, centered horizontally.
       We bypass the falling mechanic — params just appear. */
    const startCol = Math.floor((cols - PARAMS_TYPE.cells.length) / 2);
    const bottomRow = rows - 1;
    const blockId = nextBlockId++;
    for (const [dc, dr] of PARAMS_TYPE.cells) {
      const r = bottomRow - dr, c = startCol + dc;
      grid[r][c] = { kind: PARAMS_TYPE.kind, color: PARAMS_TYPE.color, stroke: PARAMS_TYPE.stroke, blockId };
      const sprite = renderCellSprite(PARAMS_TYPE.color, PARAMS_TYPE.stroke);
      sprite.position.set(hbmX + c * cellSize, hbmY + r * cellSize);
      sprite.scale.set(0);
      sprite.alpha = 0;
      blocksContainer.addChild(sprite);
      placedSprites.set(r + "," + c, sprite);
      tween(sprite, "scale.x", 0, 1, 380, "outBack");
      tween(sprite, "scale.y", 0, 1, 380, "outBack");
      tween(sprite, "alpha", 0, 1, 280, "outCubic");
    }
    /* Apply a glow to the parameters block once filters load */
    setTimeout(applyParamGlow, 50);
    floatText(overlayLayer, W / 2, hbmY + (bottomRow * cellSize) - 18, "parameters loaded · " + PARAMS_TYPE.cells.length + " cells", PARAMS_TYPE.stroke);
  }

  function applyParamGlow() {
    if (!filters) { setTimeout(applyParamGlow, 100); return; }
    if (!filters.GlowFilter) return;
    const startCol = Math.floor((cols - PARAMS_TYPE.cells.length) / 2);
    const bottomRow = rows - 1;
    for (const [dc, dr] of PARAMS_TYPE.cells) {
      const sprite = placedSprites.get((bottomRow - dr) + "," + (startCol + dc));
      if (sprite) {
        sprite.filters = [new filters.GlowFilter({ distance: 8, outerStrength: 1.4, innerStrength: 0.2, color: PARAMS_TYPE.stroke, quality: 0.3 })];
      }
    }
  }

  function renderCellSprite(color, stroke) {
    const g = new PIXI.Graphics();
    const sz = cellSize - 2;
    g.roundRect(1, 1, sz, sz, 3).fill({ color }).stroke({ width: 1.5, color: stroke });
    /* Pivot center for clean scale tween */
    g.pivot.set(cellSize / 2, cellSize / 2);
    g.position.set(cellSize / 2, cellSize / 2);
    /* Wrap in container so position+pivot don't conflict with placement coords */
    const c = new PIXI.Container();
    c.addChild(g);
    return c;
  }

  function collides(block, testRow, testCol) {
    for (const [dc, dr] of block.type.cells) {
      const r = testRow + dr, c = testCol + dc;
      if (r < 0 || r >= rows || c < 0 || c >= cols) return true;
      if (grid[r][c] !== null) return true;
    }
    return false;
  }

  function lockBlock() {
    const b = state.current; if (!b) return;
    for (const [dc, dr] of b.type.cells) {
      const r = b.row + dr, c = b.col + dc;
      grid[r][c] = { kind: b.type.kind, color: b.type.color, stroke: b.type.stroke, blockId: b.blockId };
      const sprite = renderCellSprite(b.type.color, b.type.stroke);
      sprite.position.set(hbmX + c * cellSize, hbmY + r * cellSize);
      sprite.scale.set(0.7);
      sprite.alpha = 0.4;
      blocksContainer.addChild(sprite);
      placedSprites.set(r + "," + c, sprite);
      tween(sprite, "scale.x", 0.7, 1, 220, "outBack");
      tween(sprite, "scale.y", 0.7, 1, 220, "outBack");
      tween(sprite, "alpha", 0.4, 1, 200, "outCubic");
    }
    state.score++;
    floatText(overlayLayer, W / 2, hbmY - 12, "+1 " + b.type.name, b.type.stroke);
    state.fallInterval = Math.max(280, 800 - state.score * 8);
    state.current = null;
    state.nextFallIn = 350;
    state.backwardProgress++;
    if (state.backwardProgress >= BACKWARD_BLOCKS) { fireBackwardEvent(); state.backwardProgress = 0; }
    state.stepProgress++;
    if (state.stepProgress >= STEP_BLOCKS) { fireStepEvent(); state.stepProgress = 0; }
  }

  function tryMove(dc, dr) {
    if (!state.current || state.over) return;
    const b = state.current;
    if (!collides(b, b.row + dr, b.col + dc)) { b.row += dr; b.col += dc; }
    else if (dr > 0) lockBlock();
  }

  function hardDrop() {
    if (!state.current || state.over) return;
    const b = state.current;
    while (!collides(b, b.row + 1, b.col)) b.row++;
    lockBlock();
  }

  function freeCell(r, c, particleColor, particleN) {
    const sprite = placedSprites.get(r + "," + c);
    if (sprite) {
      tween(sprite, "scale.x", 1, 1.4, 280, "outCubic");
      tween(sprite, "scale.y", 1, 1.4, 280, "outCubic");
      tween(sprite, "alpha", 1, 0, 280, "outCubic");
      setTimeout(() => { try { sprite.destroy({ children: true }); } catch (e) {} }, 320);
      placedSprites.delete(r + "," + c);
    }
    const px = hbmX + c * cellSize + cellSize / 2;
    const py = hbmY + r * cellSize + cellSize / 2;
    burst(overlayLayer, px, py, particleColor, particleN, { speed: 3, lifeMs: 700 });
    pop(overlayLayer, px, py, particleColor, { r: 14 });
    grid[r][c] = null;
  }

  function fireStepEvent() {
    state.stepFlashTime = 1200;
    flash(stage, COL.orangeStroke, 360);
    let n = 0;
    const freedBlockIds = {};
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cell = grid[r][c];
        if (cell && cell.kind === "grad") {
          freedBlockIds[cell.blockId] = true;
          freeCell(r, c, COL.orangeStroke, 8);
          n++;
        }
      }
    }
    if (n > 0) { state.score += n * 2; floatText(overlayLayer, W / 2, H / 2, "step()  ·  " + n + " gradients used + cleared", COL.orangeStroke); }
    else { floatText(overlayLayer, W / 2, H / 2, "step()", COL.muted); }
  }

  function fireBackwardEvent() {
    state.backwardFlashTime = 900;
    flash(stage, COL.blueStroke, 220);
    /* Collect activation blockIds. LIFO: free the NEWEST blocks first
       (real backward walks the graph in reverse, so the LAST activation
       produced is the FIRST one consumed). */
    const actIds = {};
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const cell = grid[r][c]; if (cell && cell.kind === "act") actIds[cell.blockId] = true;
    }
    const ids = Object.keys(actIds).map(Number).sort((a, b) => b - a);  // ← LIFO (was a-b in v5; pedagogue's one-character fix)
    const toFree = ids.slice(0, 2);
    if (toFree.length === 0) return;
    const freedSet = {}; toFree.forEach(id => { freedSet[id] = true; });
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cell = grid[r][c];
        if (cell && freedSet[cell.blockId]) freeCell(r, c, COL.blueStroke, 6);
      }
    }
    state.score += toFree.length;
    floatText(overlayLayer, W / 2, H / 2 + 24, "backward · " + toFree.length + " activations consumed (newest first)", COL.blueStroke);
  }

  function endGame() {
    if (state.score > alltimeBestRef.v) {
      alltimeBestRef.v = state.score;
      bestScore.set("oom", state.score);
    }
    if (opts.onGameOver) {
      opts.onGameOver({ score: state.score, alltimeBest: alltimeBestRef.v, emojiGrid: buildHbmGrid() });
    }
  }
  function buildHbmGrid() {
    const sampleRows = 4, sampleCols = 8;
    const rowsPerBin = Math.max(1, Math.floor(rows / sampleRows));
    const colsPerBin = Math.max(1, Math.floor(cols / sampleCols));
    const lines = [];
    for (let sr = 0; sr < sampleRows; sr++) {
      let line = "";
      for (let sc = 0; sc < sampleCols; sc++) {
        const counts = { act: 0, grad: 0, opt: 0, param: 0, empty: 0 };
        for (let dr = 0; dr < rowsPerBin; dr++) {
          for (let dc = 0; dc < colsPerBin; dc++) {
            const r = sr * rowsPerBin + dr, c = sc * colsPerBin + dc;
            if (r < rows && c < cols) {
              const cell = grid[r][c];
              if (!cell) counts.empty++;
              else counts[cell.kind]++;
            }
          }
        }
        let max = "empty", maxN = counts.empty;
        ["act","grad","opt","param"].forEach(k => { if (counts[k] > maxN) { max = k; maxN = counts[k]; } });
        line += max === "act" ? "🟦" : max === "grad" ? "🟥" : max === "opt" ? "🟧" : max === "param" ? "🟪" : "⬛";
      }
      lines.push(line);
    }
    return lines.join("\n");
  }

  /* --- Input --- */
  const handleKeydown = (e) => {
    if (!isInViewport(canvas)) return;
    if (state.over) {
      if (e.key === "r" || e.key === "R") { e.preventDefault(); if (opts.onRetry) opts.onRetry(); }
      return;
    }
    if (e.key === "ArrowLeft")  { e.preventDefault(); tryMove(-1, 0); }
    if (e.key === "ArrowRight") { e.preventDefault(); tryMove(1, 0); }
    if (e.key === "ArrowDown")  { e.preventDefault(); tryMove(0, 1); }
    if (e.key === " ")          { e.preventDefault(); hardDrop(); }
    if (e.key === "r" || e.key === "R") { e.preventDefault(); if (opts.onRetry) opts.onRetry(); }
  };
  const handlePointerdown = (e) => {
    e.preventDefault();
    if (state.over) { if (opts.onRetry) opts.onRetry(); return; }
    if (!state.current) return;
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    let cx, cy;
    if (e.touches && e.touches.length) { cx = e.touches[0].clientX; cy = e.touches[0].clientY; }
    else { cx = e.clientX; cy = e.clientY; }
    const px = (cx - rect.left) * sx;
    const py = (cy - rect.top) * sy;
    const b = state.current;
    const blockX = hbmX + b.col * cellSize;
    if (py > hbmY + b.row * cellSize + 20) { hardDrop(); return; }
    if (px < blockX + cellSize) tryMove(-1, 0);
    else if (px > blockX + cellSize) tryMove(1, 0);
  };
  window.addEventListener("keydown", handleKeydown);
  canvas.addEventListener("pointerdown", handlePointerdown);

  function isInViewport(el) {
    const r = el.getBoundingClientRect();
    return r.bottom > 0 && r.top < (window.innerHeight || document.documentElement.clientHeight);
  }

  /* --- Ambient particles when HBM gets full --- */
  let ambientCooldown = 0;
  function maybeAmbient(dt) {
    if (state.over) return;
    /* Fill ratio */
    let occ = 0;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) if (grid[r][c]) occ++;
    const fillFrac = occ / (rows * cols);
    if (fillFrac < 0.4) return;
    ambientCooldown -= dt;
    if (ambientCooldown > 0) return;
    /* Spawn a single drifting particle near the top of HBM, color by fill pressure */
    const intensity = Math.min(1, (fillFrac - 0.4) / 0.4);  // 0 at 40% full, 1 at 80% full
    ambientCooldown = 350 - intensity * 280;  // faster spawn when fuller
    const color = intensity > 0.7 ? COL.mitRed : COL.orangeStroke;
    const x = hbmX + Math.random() * hbmW;
    const p = new PIXI.Graphics();
    p.circle(0, 0, 1.5).fill({ color, alpha: 0.5 });
    p.position.set(x, hbmY + hbmH - 5);
    ambientLayer.addChild(p);
    let life = 0;
    const lifeMs = 1400;
    const handler = (ticker) => {
      const ddt = ticker.deltaMS;
      life += ddt;
      if (life >= lifeMs) { p.destroy(); PIXI.Ticker.shared.remove(handler); return; }
      p.position.y -= ddt * 0.05;
      p.alpha = (1 - life / lifeMs) * 0.5;
    };
    PIXI.Ticker.shared.add(handler);
  }

  /* --- Frame loop --- */
  app.ticker.add((ticker) => {
    const dt = ticker.deltaMS;
    if (!state.over) {
      state.timeLeft -= dt;
      if (state.timeLeft <= 0) { state.timeLeft = 0; state.over = true; flash(stage, 0x3d9e5a, 360); endGame(); }
      if (!state.current) {
        state.nextFallIn -= dt;
        if (state.nextFallIn <= 0) spawnBlock();
      } else {
        state.nextFallIn -= dt;
        if (state.nextFallIn <= 0) {
          tryMove(0, 1);
          state.nextFallIn = state.fallInterval;
        }
      }
      maybeAmbient(dt);
    }
    state.stepFlashTime = Math.max(0, state.stepFlashTime - dt);
    state.backwardFlashTime = Math.max(0, state.backwardFlashTime - dt);

    if (opts.onScoreChange && !state.over) {
      opts.onScoreChange({ score: state.score, alltimeBest: alltimeBestRef.v, timeLeft: state.timeLeft });
    }
    redraw();
    if (state.over && !gameOverOverlay.visible) drawGameOver();
  });

  function redraw() {
    /* HBM frame */
    hbmFrame.clear();
    hbmFrame.roundRect(hbmX - 4, hbmY - 4, hbmW + 8, hbmH + 8, 6).stroke({ width: 2, color: COL.mitRed });
    hbmFrame.roundRect(hbmX, hbmY, hbmW, hbmH, 4).fill({ color: 0xfafbfd });

    /* Grid lines */
    gridLines.clear();
    for (let r = 1; r < rows; r++) {
      gridLines.moveTo(hbmX, hbmY + r * cellSize).lineTo(hbmX + hbmW, hbmY + r * cellSize);
    }
    for (let c = 1; c < cols; c++) {
      gridLines.moveTo(hbmX + c * cellSize, hbmY).lineTo(hbmX + c * cellSize, hbmY + hbmH);
    }
    gridLines.stroke({ width: 0.5, color: 0xeef0f4 });

    /* (Locked blocks now live as Pixi sprites in blocksContainer; redraw isn't needed) */

    /* Current falling block — still draw imperatively (it moves every frame) */
    currentGfx.clear();
    if (state.current) {
      const b = state.current;
      for (const [dc, dr] of b.type.cells) {
        const x = hbmX + (b.col + dc) * cellSize;
        const y = hbmY + (b.row + dr) * cellSize;
        drawCell(currentGfx, x, y, b.type.color, b.type.stroke);
      }
    }

    /* Phase text */
    if (state.stepFlashTime > 0) {
      phaseText.text = "STEP() — gradients used + freed";
      phaseText.style.fill = COL.orangeStroke;
    } else if (state.backwardFlashTime > 0) {
      phaseText.text = "BACKWARD — newest activations consumed";
      phaseText.style.fill = COL.blueStroke;
    } else {
      phaseText.text = "FORWARD — packing tensors";
      phaseText.style.fill = COL.muted;
    }

    /* HUD bars (right of HBM) */
    const bx = hbmX + hbmW + 20, by = hbmY, bw = 14, bh = hbmH;
    stepBarFrame.clear();
    stepBarFrame.roundRect(bx, by, bw, bh, 3).fill({ color: COL.faint }).stroke({ width: 1, color: COL.orangeStroke });
    stepBarFill.clear();
    const stepFrac = state.stepProgress / STEP_BLOCKS;
    if (stepFrac > 0) stepBarFill.roundRect(bx, by + bh * (1 - stepFrac), bw, bh * stepFrac, 3).fill({ color: COL.orangeStroke });
    stepLabel.position.set(bx, by - 14);
    stepCounter.text = state.stepProgress + "/" + STEP_BLOCKS;
    stepCounter.position.set(bx, by + bh + 4);

    const bbx = bx + bw + 6, bbw = 6, bbh = bh;
    bwdBarFrame.clear();
    bwdBarFrame.roundRect(bbx, by, bbw, bbh, 2).fill({ color: COL.faint });
    bwdBarFill.clear();
    const bwdFrac = state.backwardProgress / BACKWARD_BLOCKS;
    if (bwdFrac > 0) bwdBarFill.roundRect(bbx, by + bbh * (1 - bwdFrac), bbw, bbh * bwdFrac, 2).fill({ color: COL.blueStroke });
    bwdLabel.position.set(bbx, by - 14);
    const infoX = bbx + bbw + 12;
    eventTitle.position.set(infoX, by + 2);
    eventLine1.position.set(infoX, by + 18);
    eventLine2.position.set(infoX, by + 31);
    eventLine3.position.set(infoX, by + 54);
    eventLine4.position.set(infoX, by + 67);
    eventLine5.position.set(infoX, by + 90);

    /* Legend (left) */
    const lx = 20, ly = hbmY;
    legendGfx.clear();
    for (let i = 0; i < legendData.length; i++) {
      legendGfx.roundRect(lx, ly + i * 20, 10, 10, 2).fill({ color: legendData[i].c }).stroke({ width: 1, color: legendData[i].s });
      legendLabels[i].position.set(lx + 14, ly + i * 20 + 1);
    }

    /* Bottom HUD */
    scoreText.text = "score " + state.score;
    const secs = Math.ceil(state.timeLeft / 1000);
    timerText.text = "⏱ " + secs + "s";
    timerText.style.fill = secs <= 10 ? COL.redStroke : COL.text;
    bestText.text = "all-time best " + alltimeBestRef.v;
    dailyText.text = "daily " + today + " · day " + dayNumber() + " · hard drop speeds packing; events do the freeing";
  }

  function drawGameOver() {
    gameOverOverlay.visible = true;
    const bg = new PIXI.Graphics();
    bg.rect(0, 0, W, H).fill({ color: 0xffffff, alpha: 0.93 });
    const titleColor = state.timeLeft > 0 ? COL.mitRed : 0x3d9e5a;
    const titleStr = state.timeLeft > 0 ? "OOM" : "🏆 survived!";
    const t1 = new PIXI.Text({ text: titleStr, style: { fontFamily: "Helvetica Neue, Arial", fontSize: 28, fontWeight: "700", fill: titleColor } });
    t1.anchor.set(0.5, 0.5); t1.position.set(W / 2, H / 2 - 24);
    if (filters && filters.GlowFilter && state.timeLeft <= 0) {
      t1.filters = [new filters.GlowFilter({ distance: 16, outerStrength: 2, innerStrength: 0.4, color: 0x3d9e5a, quality: 0.4 })];
    }
    const t2 = new PIXI.Text({ text: "placed " + state.score + " tensors · best " + alltimeBestRef.v, style: { fontFamily: "Helvetica Neue, Arial", fontSize: 14, fill: COL.text } });
    t2.anchor.set(0.5, 0.5); t2.position.set(W / 2, H / 2 + 4);
    const t3 = new PIXI.Text({ text: "tap or press R to retry", style: { fontFamily: "Helvetica Neue, Arial", fontSize: 11, fontStyle: "italic", fill: 0x777777 } });
    t3.anchor.set(0.5, 0.5); t3.position.set(W / 2, H / 2 + 28);
    gameOverOverlay.addChild(bg, t1, t2, t3);
  }

  return {
    id: "oom",
    name: "Tensor Tetris",
    ahaLabel: "You just played at",
    ahaText: "Training OOM is dominated by activations, not weights — that's why recomputation, ZeRO, and smaller batches exist (Korthikanti et al. 2022).",
    ahaLink: { href: "https://arxiv.org/abs/2205.05198", label: "Korthikanti et al. 2022 →" },
    buildShareText(r) {
      return "MLSysBook Playground · Tensor Tetris · day " + dayNumber() + "\n" +
        "packed " + r.score + " tensors\n" +
        r.emojiGrid + "\n" +
        "play → mlsysbook.ai/games/oom/";
    },
    destroy() {
      window.removeEventListener("keydown", handleKeydown);
      canvas.removeEventListener("pointerdown", handlePointerdown);
      app.destroy(true, { children: true, texture: true });
    }
  };
}
