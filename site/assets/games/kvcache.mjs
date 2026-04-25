import { mountPixiOnCanvas, flash, burst, floatText } from "./runtime.mjs";
import * as PIXI from "./vendor/pixi.min.mjs";

window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};
window.MLSP.games.kvcache = function(canvas, opts) { return mountKVCache(canvas, opts); };

export async function mountKVCache(canvas, opts = {}) {
  const { app, stage, width: W, height: H } = await mountPixiOnCanvas(canvas, { bg: 0xf8f9fa });

  let score = 0;
  let over = false;
  let timeElapsed = 0;
  const TOTAL_TIME = 50;

  const bgLayer = new PIXI.Container();
  const gridLayer = new PIXI.Container();
  const overlayLayer = new PIXI.Container();
  stage.addChild(bgLayer, gridLayer, overlayLayer);

  const colors = { req: 0x4a90c4, paged: 0x3d9e5a, empty: 0xe9ecef, border: 0xced4da, text: 0x333333 };

  // Grid setup: 12 cols, 8 rows
  const COLS = 12;
  const ROWS = 8;
  const CELL = 30;
  const PADDING = 4;
  const gridW = COLS * (CELL + PADDING);
  const gridH = ROWS * (CELL + PADDING);
  const startX = (W - gridW) / 2;
  const startY = (H - gridH) / 2 + 20;

  // grid[r][c] = block object or null
  const grid = Array.from({ length: ROWS }, () => Array(COLS).fill(null));

  // Visuals
  const cellGraphics = Array.from({ length: ROWS }, () => Array(COLS).fill(null));
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const g = new PIXI.Graphics();
      g.position.set(startX + c * (CELL + PADDING), startY + r * (CELL + PADDING));
      g.rect(0, 0, CELL, CELL).fill(colors.empty).stroke({ width: 1, color: colors.border });
      gridLayer.addChild(g);
      cellGraphics[r][c] = g;
      
      g.eventMode = "static";
      g.on("pointerdown", () => handleGridClick(r, c));
      g.on("pointerover", () => drawHover(r));
      g.on("pointerout", () => clearHover());
    }
  }

  // HUD
  const title = new PIXI.Text({ text: "KV Cache Packer", style: { fontSize: 16, fontWeight: "bold", fill: colors.text } });
  title.position.set(20, 15);
  const scoreText = new PIXI.Text({ text: "Served: 0", style: { fontSize: 14, fill: colors.text } });
  scoreText.position.set(20, 40);
  const timeText = new PIXI.Text({ text: "50s", style: { fontSize: 16, fontWeight: "bold", fill: colors.text } });
  timeText.anchor.set(1, 0); timeText.position.set(W - 20, 15);
  bgLayer.addChild(title, scoreText, timeText);

  const phaseText = new PIXI.Text({ text: "Phase 1: Contiguous", style: { fontSize: 18, fontWeight: "bold", fill: 0xc44444 } });
  phaseText.anchor.set(0.5); phaseText.position.set(W / 2, 30);
  bgLayer.addChild(phaseText);

  // Incoming Request
  let currentReqLength = generateReq();
  const reqBox = new PIXI.Graphics();
  bgLayer.addChild(reqBox);
  
  const nextLabel = new PIXI.Text({ text: "Next Request:", style: { fontSize: 14, fill: colors.text } });
  nextLabel.anchor.set(0.5); nextLabel.position.set(W / 2, startY - 40);
  bgLayer.addChild(nextLabel);
  const instructionText = new PIXI.Text({
    text: "Click a row with enough contiguous empty pages to place the incoming request.",
    style: { fontSize: 11, fill: 0x666666, fontFamily: "Helvetica Neue, Arial" }
  });
  instructionText.anchor.set(0.5);
  instructionText.position.set(W / 2, startY + gridH + 18);
  bgLayer.addChild(instructionText);

  const hoverGraphic = new PIXI.Graphics();
  gridLayer.addChild(hoverGraphic);

  let pagedMode = false;
  let pagedUnlocked = false;

  function generateReq() {
    return Math.floor(Math.random() * 4) + 2; // 2 to 5
  }

  function drawReq() {
    reqBox.clear();
    const reqW = currentReqLength * (CELL + PADDING) - PADDING;
    reqBox.roundRect(W / 2 - reqW / 2, startY - 25, reqW, CELL, 4).fill(pagedMode ? colors.paged : colors.req);
    nextLabel.text = `Next request: ${currentReqLength} KV pages`;
  }
  drawReq();

  function drawHover(r) {
    hoverGraphic.clear();
    if (over) return;
    const startC = checkContiguous(r, currentReqLength);
    const ok = pagedMode || startC !== -1;
    const x = pagedMode ? startX : startX + Math.max(0, startC) * (CELL + PADDING);
    const w = pagedMode ? gridW - PADDING : currentReqLength * (CELL + PADDING) - PADDING;
    hoverGraphic.roundRect(x, startY + r * (CELL + PADDING), w, CELL, 4)
      .fill({ color: ok ? 0x3d9e5a : 0xc44444, alpha: 0.24 })
      .stroke({ color: ok ? 0x3d9e5a : 0xc44444, width: 2, alpha: 0.7 });
  }

  function clearHover() {
    hoverGraphic.clear();
  }

  function checkContiguous(r, length) {
    let count = 0;
    for (let c = 0; c < COLS; c++) {
      if (grid[r][c] === null) {
        count++;
        if (count === length) return c - length + 1; // return start index
      } else {
        count = 0;
      }
    }
    return -1;
  }

  function handleGridClick(r, c) {
    if (over) return;

    if (!pagedMode) {
      // Phase 1: Needs contiguous space in the clicked row
      const startC = checkContiguous(r, currentReqLength);
      if (startC !== -1) {
        // Place it
        const reqId = Math.random();
        const ttl = 4 + Math.random() * 5; // 4 to 9 seconds
        for (let i = 0; i < currentReqLength; i++) {
          grid[r][startC + i] = { id: reqId, ttl, color: colors.req };
        }
        score++;
        floatText(stage, W/2, startY - 6, `served ${currentReqLength} pages`, 0x3d9e5a, { size: 12 });
        onPlaced();
      } else {
        floatText(stage, startX + c * (CELL + PADDING), startY + r * (CELL + PADDING), `Need ${currentReqLength} contiguous`, 0xc44444);
      }
    } else {
      // Phase 2: Paged mode. Click anywhere to place.
      // Shatter into available holes.
      let placed = 0;
      const reqId = Math.random();
      const ttl = 4 + Math.random() * 5;
      
      // Try to place exactly currentReqLength blocks
      // We gather all empty spots
      const empties = [];
      for (let rr = 0; rr < ROWS; rr++) {
        for (let cc = 0; cc < COLS; cc++) {
          if (grid[rr][cc] === null) empties.push({rr, cc});
        }
      }

      if (empties.length >= currentReqLength) {
        for (let i = 0; i < currentReqLength; i++) {
          const spot = empties[i];
          grid[spot.rr][spot.cc] = { id: reqId, ttl, color: colors.paged };
          burst(stage, startX + spot.cc * (CELL + PADDING) + CELL/2, startY + spot.rr * (CELL + PADDING) + CELL/2, colors.paged, 3, {speed: 1, lifeMs: 300});
        }
        score++;
        floatText(stage, W/2, startY - 6, `paged ${currentReqLength} blocks`, 0x3d9e5a, { size: 12 });
        onPlaced();
      } else {
        floatText(stage, W/2, startY - 10, "Cache Full!", 0xc44444);
      }
    }
  }

  function onPlaced() {
    scoreText.text = "Served: " + score;
    if (opts.onScoreChange) opts.onScoreChange({ score, timeLeft: Math.max(0, TOTAL_TIME - timeElapsed) });
    currentReqLength = generateReq();
    clearHover();
    drawReq();
  }

  function defrag() {
    if (!pagedMode || over) return;
    for (let r = 0; r < ROWS; r++) {
      let writeC = 0;
      for (let readC = 0; readC < COLS; readC++) {
        if (grid[r][readC] !== null) {
          if (writeC !== readC) {
            grid[r][writeC] = grid[r][readC];
            grid[r][readC] = null;
          }
          writeC++;
        }
      }
    }
    floatText(stage, W/2, H - 30, "Defragged!", 0x3d9e5a);
  }

  const handleKeydown = (e) => {
    if (e.code === "Space") { e.preventDefault(); defrag(); }
    if (over && (e.key === "r" || e.key === "R")) {
      if (opts.onRetry) opts.onRetry();
    }
  };
  window.addEventListener("keydown", handleKeydown);

  app.ticker.add((ticker) => {
    const dt = ticker.deltaMS;
    if (over) return;

    timeElapsed += dt / 1000;
    const timeLeft = Math.max(0, TOTAL_TIME - timeElapsed);
    timeText.text = Math.ceil(timeLeft) + "s";

    if (timeElapsed >= 20 && !pagedUnlocked) {
      pagedUnlocked = true;
      pagedMode = true;
      phaseText.text = "Phase 2: PAGED MODE UNLOCKED";
      instructionText.text = "Paged mode: click anywhere with enough free pages. Press Space to compact rows.";
      phaseText.style.fill = 0x3d9e5a;
      flash(stage, 0x3d9e5a, 600);
      floatText(stage, W/2, H/2, "PAGED MODE!", 0x3d9e5a, {size: 32});
      drawReq();
    }

    if (timeElapsed >= TOTAL_TIME) {
      over = true;
      drawGameOver();
      return;
    }

    // Tick blocks
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const block = grid[r][c];
        if (block) {
          block.ttl -= dt / 1000;
          if (block.ttl <= 0) {
            grid[r][c] = null;
            burst(stage, startX + c * (CELL + PADDING) + CELL/2, startY + r * (CELL + PADDING) + CELL/2, block.color, 5);
          }
        }
        
        // Update visuals
        const g = cellGraphics[r][c];
        g.clear();
        if (grid[r][c]) {
          g.rect(0, 0, CELL, CELL).fill(grid[r][c].color).stroke({ width: 1, color: 0x333333 });
        } else {
          g.rect(0, 0, CELL, CELL).fill(colors.empty).stroke({ width: 1, color: colors.border });
        }
      }
    }
  });

  function drawGameOver() {
    if (opts.onGameOver) opts.onGameOver({ score, timeLeft: 0 });
    const bg = new PIXI.Graphics();
    bg.rect(0, 0, W, H).fill({ color: 0xffffff, alpha: 0.9 });
    const t1 = new PIXI.Text({ text: "Time's Up!", style: { fontSize: 28, fontWeight: "bold", fill: 0x3d9e5a } });
    t1.anchor.set(0.5); t1.position.set(W / 2, H / 2 - 20);
    const t2 = new PIXI.Text({ text: `Served: ${score} Requests`, style: { fontSize: 16, fill: colors.text } });
    t2.anchor.set(0.5); t2.position.set(W / 2, H / 2 + 15);
    const t3 = new PIXI.Text({ text: "Tap or press R to retry", style: { fontSize: 12, fill: 0x888888, fontStyle: "italic" } });
    t3.anchor.set(0.5); t3.position.set(W / 2, H / 2 + 40);
    overlayLayer.addChild(bg, t1, t2, t3);
  }

  return {
    id: "kvcache",
    name: "KV Cache Packer",
    ahaLabel: "PagedAttention",
    ahaText: "Standard KV caching needs contiguous memory blocks, leading to high fragmentation and Out-Of-Memory errors. Paged mode splits memory into non-contiguous blocks, virtually eliminating fragmentation and enabling larger batch sizes.",
    destroy() {
      window.removeEventListener("keydown", handleKeydown);
      app.destroy(true, { children: true, texture: true });
    }
  };
}

window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};
window.MLSP.games.kvcache = mountKVCache;
