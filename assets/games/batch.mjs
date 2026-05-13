import { mountPixiOnCanvas, flash, shake, mountReadyOverlay } from "./runtime.mjs";
import * as PIXI from "./vendor/pixi.min.mjs";

export async function mountBatch(canvas, opts = {}) {
  const { app, stage, width: W, height: H } = await mountPixiOnCanvas(canvas, { bg: 0xf8f9fa });

  let score = 0;
  let over = false;
  let started = false;
  let timeLeft = 30000;
  
  let queue = 10;
  let B = 1;
  const MAX_B = 20;
  const MAX_MEM = 100;
  let timeElapsed = 0;

  const bgLayer = new PIXI.Container();
  const hudLayer = new PIXI.Container();
  const overlayLayer = new PIXI.Container();
  stage.addChild(bgLayer, hudLayer, overlayLayer);

  const colors = { q: 0x4a90c4, proc: 0x3d9e5a, mem: 0xc44444, text: 0x333333, ui: 0xe9ecef };

  // HUD
  const title = new PIXI.Text({ text: "Batch Size Balancer", style: { fontSize: 16, fontWeight: "bold", fill: colors.text } });
  title.position.set(20, 15);
  const scoreText = new PIXI.Text({ text: "Processed: 0", style: { fontSize: 14, fill: colors.text } });
  scoreText.position.set(20, 40);
  const timeText = new PIXI.Text({ text: "30s", style: { fontSize: 16, fontWeight: "bold", fill: colors.text } });
  timeText.anchor.set(1, 0); timeText.position.set(W - 20, 15);

  // Layout
  const centerX = W / 2;
  const centerY = H / 2;

  // Visualizing queue
  const qBox = new PIXI.Graphics();
  bgLayer.addChild(qBox);
  const qLabel = new PIXI.Text({ text: "Incoming Queue", style: { fontSize: 12, fill: 0x6c757d }});
  qLabel.anchor.set(0.5); qLabel.position.set(centerX - 100, centerY + 80);
  bgLayer.addChild(qLabel);

  // Visualizing batch size
  const bBox = new PIXI.Graphics();
  bBox.roundRect(centerX + 50, centerY - 40, 60, 80, 8).fill(colors.ui).stroke({ width: 2, color: 0xadb5bd });
  bgLayer.addChild(bBox);
  const bText = new PIXI.Text({ text: "1", style: { fontSize: 24, fontWeight: "bold", fill: colors.text } });
  bText.anchor.set(0.5); bText.position.set(centerX + 80, centerY);
  bgLayer.addChild(bText);
  const bLabel = new PIXI.Text({ text: "Batch Size\n(Up/Down)", align: "center", style: { fontSize: 11, fill: 0x6c757d }});
  bLabel.anchor.set(0.5); bLabel.position.set(centerX + 80, centerY + 60);
  bgLayer.addChild(bLabel);

  // Visualizing Memory
  const memFrame = new PIXI.Graphics();
  memFrame.roundRect(centerX - 150, H - 40, 300, 16, 8).stroke({ width: 2, color: 0xced4da });
  const memFill = new PIXI.Graphics();
  hudLayer.addChild(memFrame, memFill);
  const memLabel = new PIXI.Text({ text: "Memory Usage (Queue + Batch)", style: { fontSize: 11, fill: 0x6c757d }});
  memLabel.anchor.set(0.5); memLabel.position.set(centerX, H - 50);
  hudLayer.addChild(memLabel);

  const setBatch = (newB) => {
    if (!started || over) return;
    B = Math.max(1, Math.min(newB, MAX_B));
    bText.text = B.toString();
  };

  const onKey = (e) => {
    if (e.key === "ArrowUp") { e.preventDefault(); setBatch(B + 1); }
    if (e.key === "ArrowDown") { e.preventDefault(); setBatch(B - 1); }
  };
  window.addEventListener("keydown", onKey);

  // Click controls for mobile
  const btnUp = new PIXI.Graphics().roundRect(centerX + 65, centerY - 70, 30, 20, 4).fill(0xdee2e6);
  btnUp.eventMode = "static"; btnUp.on("pointerdown", () => setBatch(B + 1));
  const btnUpText = new PIXI.Text({ text: "▲", style: { fontSize: 12, fill: 0x495057 }});
  btnUpText.anchor.set(0.5); btnUpText.position.set(centerX + 80, centerY - 60);
  const btnDown = new PIXI.Graphics().roundRect(centerX + 65, centerY + 90, 30, 20, 4).fill(0xdee2e6);
  btnDown.eventMode = "static"; btnDown.on("pointerdown", () => setBatch(B - 1));
  const btnDownText = new PIXI.Text({ text: "▼", style: { fontSize: 12, fill: 0x495057 }});
  btnDownText.anchor.set(0.5); btnDownText.position.set(centerX + 80, centerY + 100);
  hudLayer.addChild(btnUp, btnUpText, btnDown, btnDownText);

  // Pre-game READY overlay
  mountReadyOverlay(stage, {
    width: W, height: H,
    title: "BATCH SIZE BALANCER",
    goal: "Push batch size up for throughput — but don't OOM.",
    controls: "↑ ↓  tune batch size · TAP arrows · R  retry",
    onLaunch: () => { started = true; }
  });

  app.ticker.add((ticker) => {
    const dt = ticker.deltaMS;
    if (!started) return;
    if (over) return;

    timeLeft -= dt;
    if (timeLeft <= 0) {
      timeLeft = 0;
      over = true;
      flash(stage, 0x3d9e5a, 400);
      drawGameOver(true);
    }
    timeText.text = Math.ceil(timeLeft / 1000) + "s";
    timeElapsed += dt / 1000;

    // Game logic
    const dtSec = dt / 1000;
    const spawnRate = 5 + Math.sin(timeElapsed * 1.5) * 4 + timeElapsed * 0.2; 
    queue += spawnRate * dtSec;

    const processRate = B * 1.5;
    const processed = Math.min(queue, processRate * dtSec);
    queue -= processed;
    score += processed;
    scoreText.text = "Processed: " + Math.floor(score);
    if (opts.onScoreChange) opts.onScoreChange({ score: Math.floor(score), timeLeft });

    const memory = queue + B * 3; // queue items + batch items cost memory
    
    if (memory > MAX_MEM) {
      over = true;
      flash(stage, colors.mem, 400);
      shake(stage, 12, 400);
      drawGameOver(false);
    }

    // Visuals update
    qBox.clear();
    const qHeight = Math.min(100, (queue / 50) * 100);
    qBox.roundRect(centerX - 130, centerY + 50 - Math.max(4, qHeight), 60, Math.max(4, qHeight), 4).fill(colors.q);

    memFill.clear();
    const memFrac = Math.min(1, memory / MAX_MEM);
    if (memFrac > 0) {
      memFill.roundRect(centerX - 150, H - 40, 300 * memFrac, 16, 8).fill(memFrac > 0.8 ? colors.mem : 0x868e96);
    }
  });

  function drawGameOver(success) {
    if (opts.onGameOver) opts.onGameOver({ score: Math.floor(score), timeLeft });
    const bg = new PIXI.Graphics();
    bg.rect(0, 0, W, H).fill({ color: 0xffffff, alpha: 0.9 });
    const titleColor = success ? 0x3d9e5a : colors.mem;
    const titleStr = success ? "Time's Up!" : "OOM Crash!";
    const t1 = new PIXI.Text({ text: titleStr, style: { fontSize: 28, fontWeight: "bold", fill: titleColor } });
    t1.anchor.set(0.5); t1.position.set(W / 2, H / 2 - 20);
    const t2 = new PIXI.Text({ text: `Processed: ${Math.floor(score)}`, style: { fontSize: 16, fill: colors.text } });
    t2.anchor.set(0.5); t2.position.set(W / 2, H / 2 + 15);
    const t3 = new PIXI.Text({ text: "Tap or press R to retry", style: { fontSize: 12, fill: 0x888888, fontStyle: "italic" } });
    t3.anchor.set(0.5); t3.position.set(W / 2, H / 2 + 40);
    overlayLayer.addChild(bg, t1, t2, t3);
  }

  const handleKeydown = (e) => {
    if (over && (e.key === "r" || e.key === "R")) {
      if (opts.onRetry) opts.onRetry();
    }
  };
  const handlePointerdown = () => {
    if (over && opts.onRetry) opts.onRetry();
  };
  window.addEventListener("keydown", handleKeydown);
  canvas.addEventListener("pointerdown", handlePointerdown);

  return {
    id: "batch",
    name: "Batch Size Balancer",
    ahaLabel: "Batch Size Tradeoff",
    ahaText: "Larger batch sizes increase throughput but also increase memory requirements. You have to balance it against incoming traffic and available memory.",
    destroy() {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keydown", handleKeydown);
      canvas.removeEventListener("pointerdown", handlePointerdown);
      app.destroy(true, { children: true, texture: true });
    }
  };
}

window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};
window.MLSP.games.batch = mountBatch;
