import { mountPixiOnCanvas, flash, floatText, shake, mountReadyOverlay } from "./runtime.mjs";
import * as PIXI from "./vendor/pixi.min.mjs";

window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};
window.MLSP.games.cluster = function(canvas, opts) { return mountCluster(canvas, opts); };

export async function mountCluster(canvas, opts = {}) {
  const { app, stage, width: W, height: H } = await mountPixiOnCanvas(canvas, { bg: 0xf8f9fa });

  let score = 0;
  let over = false;
  let started = false;
  let timeElapsed = 0;
  const TOTAL_TIME = 45;

  const bgLayer = new PIXI.Container();
  const gridLayer = new PIXI.Container();
  const hudLayer = new PIXI.Container();
  const overlayLayer = new PIXI.Container();
  stage.addChild(bgLayer, gridLayer, hudLayer, overlayLayer);

  const colors = { empty: 0xe9ecef, border: 0xced4da, text: 0x333333,
                   job1: 0x4a90c4, job2: 0xf39c12, job4: 0x9b59b6, alarm: 0xc44444 };

  const GRID_SIZE = 8;
  const CELL = 35;
  const PADDING = 4;
  const gridW = GRID_SIZE * (CELL + PADDING);
  const gridH = GRID_SIZE * (CELL + PADDING);
  const startX = (W - gridW) / 2 + 40;
  const startY = (H - gridH) / 2 + 20;

  // grid[r][c] = job object or null
  const grid = Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(null));
  const cellGraphics = Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(null));

  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      const g = new PIXI.Graphics();
      g.position.set(startX + c * (CELL + PADDING), startY + r * (CELL + PADDING));
      g.rect(0, 0, CELL, CELL).fill(colors.empty).stroke({ width: 1, color: colors.border });
      gridLayer.addChild(g);
      cellGraphics[r][c] = g;
    }
  }

  // Canvas-level pointer handlers compute the cell from coordinates. Pixi v8's
  // per-Graphics hit testing was unreliable here (clicks fired pointerover but
  // not pointerdown), so we route all grid pointer events through the canvas DOM
  // element and translate to grid (r, c) ourselves. Robust regardless of Pixi
  // event-system quirks.
  function cellAtCanvasCoord(canvasX, canvasY) {
    const c = Math.floor((canvasX - startX) / (CELL + PADDING));
    const r = Math.floor((canvasY - startY) / (CELL + PADDING));
    if (r < 0 || r >= GRID_SIZE || c < 0 || c >= GRID_SIZE) return null;
    // Reject clicks in the inter-cell padding gap.
    const localX = (canvasX - startX) - c * (CELL + PADDING);
    const localY = (canvasY - startY) - r * (CELL + PADDING);
    if (localX > CELL || localY > CELL) return null;
    return { r, c };
  }
  function pointerToCanvas(e) {
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width  / rect.width;
    const sy = canvas.height / rect.height;
    return { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy };
  }
  const onCanvasMove = (e) => {
    if (!started || over) return;
    const p = pointerToCanvas(e);
    const cell = cellAtCanvasCoord(p.x, p.y);
    if (cell) drawHover(cell.r, cell.c);
    else clearHover();
  };
  const onCanvasDown = (e) => {
    if (!started || over) return;
    const p = pointerToCanvas(e);
    const cell = cellAtCanvasCoord(p.x, p.y);
    if (cell) handleGridClick(cell.r, cell.c);
  };
  canvas.addEventListener("pointermove", onCanvasMove);
  canvas.addEventListener("pointerdown", onCanvasDown);
  canvas.addEventListener("pointerleave", () => clearHover());

  // HUD
  const title = new PIXI.Text({ text: "Cluster Commander", style: { fontSize: 16, fontWeight: "bold", fill: colors.text } });
  title.position.set(20, 15);
  const scoreText = new PIXI.Text({ text: "Scheduled: 0", style: { fontSize: 14, fill: colors.text } });
  scoreText.position.set(20, 40);
  const timeText = new PIXI.Text({ text: "45s", style: { fontSize: 16, fontWeight: "bold", fill: colors.text } });
  timeText.anchor.set(1, 0); timeText.position.set(W - 20, 15);
  hudLayer.addChild(title, scoreText, timeText);

  // Queue visual
  const queueLabel = new PIXI.Text({ text: "Job Queue", style: { fontSize: 14, fill: colors.text, fontWeight: "bold" } });
  queueLabel.position.set(20, 100);
  hudLayer.addChild(queueLabel);
  const instructionText = new PIXI.Text({
    text: "Place the first queued job on the grid. Large jobs need contiguous empty space.",
    style: { fontSize: 12, fill: 0x666666, fontFamily: "Helvetica Neue, Arial" }
  });
  instructionText.position.set(20, 72);
  hudLayer.addChild(instructionText);

  const queueBox = new PIXI.Graphics();
  queueBox.position.set(20, 130);
  hudLayer.addChild(queueBox);

  // Jobs: size = 1 (1x1), 2 (2x2), 4 (4x4)
  const jobTypes = [
    { size: 1, color: colors.job1, name: "Inference", prob: 0.6 },
    { size: 2, color: colors.job2, name: "Fine-tune", prob: 0.3 },
    { size: 4, color: colors.job4, name: "Pre-train", prob: 0.1 }
  ];

  let queue = [];
  function populateQueue() {
    while (queue.length < 5) {
      const r = Math.random();
      let type = jobTypes[0];
      if (r > 0.9) type = jobTypes[2];
      else if (r > 0.6) type = jobTypes[1];
      queue.push({ ...type, ttl: 5 + type.size * 3 }); // Larger jobs take longer
    }
    drawQueue();
  }
  populateQueue();

  function drawQueue() {
    queueBox.removeChildren();
    queueBox.clear();
    let qy = 0;
    for (let i = 0; i < queue.length; i++) {
      const job = queue[i];
      const s = job.size * 10;
      const x = i === 0 ? 0 : 8;
      queueBox.roundRect(x, qy, s, s, 3).fill(job.color).stroke({ width: i === 0 ? 2 : 1, color: i === 0 ? 0x333333 : 0xffffff });
      if (i === 0) {
        const label = new PIXI.Text({
          text: `${job.name} ${job.size}x${job.size}`,
          style: { fontSize: 11, fill: colors.text, fontFamily: "Helvetica Neue, Arial", fontWeight: "700" }
        });
        label.position.set(s + 8, qy - 1);
        queueBox.addChild(label);
      }
      qy += Math.max(s, 18) + 12;
    }
  }

  let hoverGraphic = new PIXI.Graphics();
  gridLayer.addChild(hoverGraphic);

  function canFit(r, c, size) {
    if (r + size > GRID_SIZE || c + size > GRID_SIZE) return false;
    for (let ir = r; ir < r + size; ir++) {
      for (let ic = c; ic < c + size; ic++) {
        if (grid[ir][ic] !== null) return false;
      }
    }
    return true;
  }

  function drawHover(r, c) {
    hoverGraphic.clear();
    if (over) return;
    const job = queue[0];
    const fit = canFit(r, c, job.size);
    const sizePx = job.size * (CELL + PADDING) - PADDING;
    hoverGraphic.rect(startX + c * (CELL + PADDING), startY + r * (CELL + PADDING), sizePx, sizePx)
                .fill({ color: fit ? 0x3d9e5a : 0xc44444, alpha: 0.3 });
  }

  function clearHover() {
    hoverGraphic.clear();
  }

  let alarmActive = false;
  let alarmTimer = 0;

  function handleGridClick(r, c) {
    if (!started || over) return;
    const job = queue[0];
    if (canFit(r, c, job.size)) {
      // Place job
      const reqId = Math.random();
      for (let ir = r; ir < r + job.size; ir++) {
        for (let ic = c; ic < c + job.size; ic++) {
          grid[ir][ic] = { id: reqId, ttl: job.ttl, color: job.color };
        }
      }
      queue.shift();
      score += job.size * job.size; // Score proportional to size
      scoreText.text = "Scheduled: " + score;
      floatText(stage, startX + c * (CELL + PADDING) + 18, startY + r * (CELL + PADDING) - 12, `scheduled ${job.name}`, 0x3d9e5a, { size: 12 });
      if (opts.onScoreChange) opts.onScoreChange({ score, timeLeft: Math.max(0, TOTAL_TIME - timeElapsed) });
      populateQueue();
      clearHover();
      alarmActive = false; // Reset alarm if we could schedule
    } else {
      floatText(stage, startX + c * (CELL + PADDING), startY + r * (CELL + PADDING), `Need ${job.size}x${job.size}`, colors.alarm);
    }
  }

  // Pre-game READY overlay — game starts paused so the player can read.
  mountReadyOverlay(stage, {
    width: W, height: H,
    title: "CLUSTER COMMANDER",
    goal: "Schedule jobs without fragmenting the fleet.",
    controls: "CLICK a cell to place the front-of-queue job · R  retry",
    onLaunch: () => { started = true; }
  });

  app.ticker.add((ticker) => {
    const dt = ticker.deltaMS;
    if (!started) return;
    if (over) return;

    timeElapsed += dt / 1000;
    const timeLeft = Math.max(0, TOTAL_TIME - timeElapsed);
    timeText.text = Math.ceil(timeLeft) + "s";

    if (timeElapsed >= TOTAL_TIME) {
      over = true;
      drawGameOver();
      return;
    }

    // Tick grid
    for (let r = 0; r < GRID_SIZE; r++) {
      for (let c = 0; c < GRID_SIZE; c++) {
        const job = grid[r][c];
        if (job) {
          job.ttl -= dt / 1000;
          if (job.ttl <= 0) grid[r][c] = null;
        }
      }
    }

    // Check if head of queue is permanently blocked (needs 4x4, but no 4x4 space)
    const headJob = queue[0];
    let canScheduleHead = false;
    for (let r = 0; r <= GRID_SIZE - headJob.size; r++) {
      for (let c = 0; c <= GRID_SIZE - headJob.size; c++) {
        if (canFit(r, c, headJob.size)) {
          canScheduleHead = true;
          break;
        }
      }
      if (canScheduleHead) break;
    }

    if (!canScheduleHead) {
      alarmActive = true;
      alarmTimer += dt;
      if (alarmTimer > 500) {
        alarmTimer = 0;
        // Flashing effect on queue head
        queueLabel.style.fill = queueLabel.style.fill === colors.alarm ? colors.text : colors.alarm;
        if (Math.random() > 0.7) shake(stage, 2, 100);
      }
    } else {
      alarmActive = false;
      queueLabel.style.fill = colors.text;
    }

    // Draw grid
    for (let r = 0; r < GRID_SIZE; r++) {
      for (let c = 0; c < GRID_SIZE; c++) {
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

  const handleKeydown = (e) => {
    if (over && (e.key === "r" || e.key === "R")) {
      if (opts.onRetry) opts.onRetry();
    }
  };
  window.addEventListener("keydown", handleKeydown);

  function drawGameOver() {
    if (opts.onGameOver) opts.onGameOver({ score, timeLeft: 0 });
    const bg = new PIXI.Graphics();
    bg.rect(0, 0, W, H).fill({ color: 0xffffff, alpha: 0.9 });
    const t1 = new PIXI.Text({ text: "Time's Up!", style: { fontSize: 28, fontWeight: "bold", fill: 0x3d9e5a } });
    t1.anchor.set(0.5); t1.position.set(W / 2, H / 2 - 20);
    const t2 = new PIXI.Text({ text: `Scheduled: ${score} compute units`, style: { fontSize: 16, fill: colors.text } });
    t2.anchor.set(0.5); t2.position.set(W / 2, H / 2 + 15);
    const t3 = new PIXI.Text({ text: "Tap or press R to retry", style: { fontSize: 12, fill: 0x888888, fontStyle: "italic" } });
    t3.anchor.set(0.5); t3.position.set(W / 2, H / 2 + 40);
    overlayLayer.addChild(bg, t1, t2, t3);
  }

  return {
    id: "cluster",
    name: "Cluster Commander",
    ahaLabel: "Fleet Fragmentation",
    ahaText: "When small jobs scatter across a cluster, they cause fragmentation. Even if 50% of the GPUs are idle, a massive pre-training job requiring contiguous nodes will be blocked, ruining cluster utilization.",
    destroy() {
      window.removeEventListener("keydown", handleKeydown);
      canvas.removeEventListener("pointermove", onCanvasMove);
      canvas.removeEventListener("pointerdown", onCanvasDown);
      app.destroy(true, { children: true, texture: true });
    }
  };
}

window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};
window.MLSP.games.cluster = mountCluster;
