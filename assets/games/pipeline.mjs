import { mountPixiOnCanvas, burst, flash, shake, mountReadyOverlay } from "./runtime.mjs";
import * as PIXI from "./vendor/pixi.min.mjs";

export async function mountPipeline(canvas, opts = {}) {
  const { app, stage, width: W, height: H } = await mountPixiOnCanvas(canvas, { bg: 0xffffff });

  let score = 0;
  let activeBlocks = [];
  let over = false;
  let started = false;             // pre-game READY gate
  let timeLeft = 30000;
  const MAX_MEMORY = 8;
  let nextBlockId = 0;

  // Wider stages so the GPUs fill more of the canvas — was 82px, leaving ~175px
  // of dead space on the right. Now 116px so the four GPUs span ~464px and the
  // right-side gap drops to ~50px (matching the left margin).
  const stageWidth = 116;
  const stageHeight = 124;
  const startX = W / 2 - 2 * stageWidth;
  const startY = 132;
  const TURN_MS = 600;             // smooth blue→brown turnaround duration

  const bgLayer = new PIXI.Container();
  const blockLayer = new PIXI.Container();
  const hudLayer = new PIXI.Container();
  const overlayLayer = new PIXI.Container();
  stage.addChild(bgLayer, blockLayer, hudLayer, overlayLayer);

  const colors = { fwd: 0x4a90c4, bwd: 0xc87b2a, mem: 0xc44444, box: 0xffffff, text: 0x333333, rail: 0xd8e4ee };

  const rail = new PIXI.Graphics();
  rail.roundRect(startX - 26, startY + 20, stageWidth * 4 + 52, 32, 16).fill({ color: 0xebf3fa });
  rail.roundRect(startX - 26, startY + stageHeight - 52, stageWidth * 4 + 52, 32, 16).fill({ color: 0xfdf1e2 });
  rail.moveTo(startX - 12, startY + 36).lineTo(startX + stageWidth * 4 + 12, startY + 36).stroke({ color: colors.fwd, width: 2, alpha: 0.42 });
  rail.moveTo(startX + stageWidth * 4 + 12, startY + stageHeight - 36).lineTo(startX - 12, startY + stageHeight - 36).stroke({ color: colors.bwd, width: 2, alpha: 0.42 });
  bgLayer.addChild(rail);

  // Draw pipeline stages as clean device cards.
  for (let i = 0; i < 4; i++) {
    const box = new PIXI.Graphics();
    box.roundRect(startX + i * stageWidth + 5, startY, stageWidth - 10, stageHeight, 10)
      .fill(colors.box)
      .stroke({ width: 1.5, color: 0xd7dde5 });
    box.rect(startX + i * stageWidth + 5, startY, stageWidth - 10, 5).fill({ color: i === 0 ? 0xa31f34 : 0x4a90c4, alpha: 0.9 });
    bgLayer.addChild(box);
    const text = new PIXI.Text({ text: `GPU ${i}`, style: { fontSize: 12, fill: 0x555555, fontFamily: "Helvetica Neue, Arial", fontWeight: "bold" }});
    text.anchor.set(0.5);
    text.position.set(startX + i * stageWidth + stageWidth / 2, startY + 18);
    bgLayer.addChild(text);
  }

  const legend = new PIXI.Container();
  const legendBg = new PIXI.Graphics();
  legendBg.roundRect(0, 0, 184, 48, 8).fill({ color: 0xffffff, alpha: 0.92 }).stroke({ color: 0xe0e0e0, width: 1 });
  const fwdSwatch = new PIXI.Graphics();
  fwdSwatch.roundRect(12, 12, 22, 5, 2).fill({ color: colors.fwd });
  const bwdSwatch = new PIXI.Graphics();
  bwdSwatch.roundRect(12, 31, 22, 5, 2).fill({ color: colors.bwd });
  const fwdLabel = new PIXI.Text({ text: "forward activations", style: { fontSize: 10, fill: 0x555555, fontFamily: "Helvetica Neue, Arial", fontWeight: "700" }});
  const bwdLabel = new PIXI.Text({ text: "backward gradients", style: { fontSize: 10, fill: 0x555555, fontFamily: "Helvetica Neue, Arial", fontWeight: "700" }});
  fwdLabel.position.set(42, 8);
  bwdLabel.position.set(42, 27);
  legend.position.set(20, 88);
  legend.addChild(legendBg, fwdSwatch, bwdSwatch, fwdLabel, bwdLabel);
  bgLayer.addChild(legend);

  // HUD
  const title = new PIXI.Text({ text: "Pipeline Pacer", style: { fontSize: 16, fontWeight: "bold", fill: colors.text } });
  title.position.set(20, 15);
  
  const scoreText = new PIXI.Text({ text: "Score: 0", style: { fontSize: 14, fill: colors.text } });
  scoreText.position.set(20, 40);
  
  const timeText = new PIXI.Text({ text: "30s", style: { fontSize: 16, fontWeight: "bold", fill: colors.text } });
  timeText.anchor.set(1, 0);
  timeText.position.set(W - 20, 15);

  const memText = new PIXI.Text({ text: "Memory (Active Microbatches)", style: { fontSize: 11, fill: 0x6c757d } });
  memText.anchor.set(0.5);
  memText.position.set(W / 2, H - 45);

  const memFrame = new PIXI.Graphics();
  memFrame.roundRect(W / 2 - 100, H - 30, 200, 14, 7).stroke({ width: 2, color: 0xced4da });
  const memFill = new PIXI.Graphics();

  const hintText = new PIXI.Text({ text: "Tap or Space to admit the next microbatch", style: { fontSize: 12, fill: 0x777777, fontStyle: "italic" } });
  hintText.anchor.set(0.5);
  hintText.position.set(W / 2, 60);

  hudLayer.addChild(title, scoreText, timeText, memText, memFrame, memFill, hintText);

  // Pre-game READY overlay — every game now starts paused so the player can read.
  mountReadyOverlay(stage, {
    width: W, height: H,
    title: "PIPELINE PACER",
    goal: "Keep the GPUs fed. Hide bubbles, but don't OOM.",
    controls: "SPACE  admit a microbatch · TAP  same",
    onLaunch: () => { started = true; }
  });

  // Smooth color crossfade for the U-turn at the last GPU. Blends RGB so the
  // sprite fades from compute-blue into routing-orange instead of snapping.
  function lerpColor(c1, c2, t) {
    const r1 = (c1 >> 16) & 0xff, g1 = (c1 >> 8) & 0xff, b1 = c1 & 0xff;
    const r2 = (c2 >> 16) & 0xff, g2 = (c2 >> 8) & 0xff, b2 = c2 & 0xff;
    return ((Math.round(r1 + (r2 - r1) * t) << 16) |
            (Math.round(g1 + (g2 - g1) * t) << 8)  |
             Math.round(b1 + (b2 - b1) * t));
  }

  function spawn() {
    if (!started || over) return;
    if (activeBlocks.length >= MAX_MEMORY) {
      over = true;
      flash(stage, colors.mem, 400);
      shake(stage, 12, 400);
      drawGameOver(false);
      return;
    }
    const block = {
      id: nextBlockId++,
      isFwd: true,
      stageIndex: 0,
      progress: 0,
      turning: false,        // U-turn animation state (the "fade blue → brown" moment)
      turnT: 0,
      sprite: new PIXI.Graphics()
    };
    block.sprite.roundRect(-13, -13, 26, 26, 6).fill(colors.fwd).stroke({ color: 0x2f6e9d, width: 1 });
    blockLayer.addChild(block.sprite);
    activeBlocks.push(block);
  }

  const onDown = (e) => { e.preventDefault(); spawn(); };
  canvas.addEventListener("pointerdown", onDown);
  const onKey = (e) => { if (e.code === "Space") { e.preventDefault(); spawn(); } };
  window.addEventListener("keydown", onKey);

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

    const SPEED = 0.0015; // Time per stage
    for (let i = activeBlocks.length - 1; i >= 0; i--) {
      const b = activeBlocks[i];

      // Turn-around animation: hold under GPU 3, fade blue → brown, U-curve down.
      if (b.turning) {
        b.turnT += dt / TURN_MS;
        const t = Math.min(1, b.turnT);
        const eased = 1 - Math.pow(1 - t, 3);
        const blended = lerpColor(colors.fwd, colors.bwd, eased);
        // Vertical U-curve: down then back up to the lower (backward) rail.
        const cx = startX + 3 * stageWidth + stageWidth / 2;
        const baseY = startY + stageHeight / 2;
        const dipY  = baseY + 26 + Math.sin(t * Math.PI) * 18;     // dip in the middle of the turn
        b.sprite.clear()
          .roundRect(-13, -13, 26, 26, 6)
          .fill({ color: blended })
          .stroke({ color: lerpColor(0x2f6e9d, 0x9a6620, eased), width: 1 });
        b.sprite.position.set(cx, baseY - 26 + (dipY - (baseY - 26)) * eased);
        if (t >= 1) {
          b.turning = false;
          b.isFwd = false;
          b.stageIndex = 3;
          b.progress = 0;
          burst(blockLayer, b.sprite.x, b.sprite.y, colors.bwd, 4, { speed: 1.5, lifeMs: 300 });
        }
        continue;
      }

      b.progress += dt * SPEED;

      if (b.progress >= 1) {
        b.progress = 0;
        if (b.isFwd) {
          b.stageIndex++;
          if (b.stageIndex >= 4) {
            // Enter U-turn instead of instant color swap. Smooth blue → brown.
            b.turning = true;
            b.turnT = 0;
            continue;
          }
        } else {
          b.stageIndex--;
          if (b.stageIndex < 0) {
            burst(blockLayer, b.sprite.x, b.sprite.y, 0x3d9e5a, 6);
            b.sprite.destroy();
            activeBlocks.splice(i, 1);
            score++;
            scoreText.text = "Score: " + score;
            if (opts.onScoreChange) opts.onScoreChange({ score, timeLeft });
            continue;
          }
        }
      }

      let currentX = startX + b.stageIndex * stageWidth + stageWidth / 2;
      let nextIdx = b.isFwd ? b.stageIndex + 1 : b.stageIndex - 1;
      let targetX = startX + nextIdx * stageWidth + stageWidth / 2;
      if (nextIdx > 3 || nextIdx < 0) targetX = currentX + (b.isFwd ? stageWidth : -stageWidth);

      let x = currentX + (targetX - currentX) * b.progress;
      let yOffset = b.isFwd ? -26 : 26;
      b.sprite.position.set(x, startY + stageHeight / 2 + yOffset);
    }

    memFill.clear();
    const memFrac = activeBlocks.length / MAX_MEMORY;
    if (memFrac > 0) {
      const barColor = memFrac >= 0.8 ? colors.mem : 0x868e96;
      memFill.roundRect(W / 2 - 100, H - 30, 200 * memFrac, 14, 7).fill(barColor);
    }
  });

  function drawGameOver(success) {
    if (opts.onGameOver) opts.onGameOver({ score, timeLeft });
    const bg = new PIXI.Graphics();
    bg.rect(0, 0, W, H).fill({ color: 0xffffff, alpha: 0.9 });
    const titleColor = success ? 0x3d9e5a : colors.mem;
    const titleStr = success ? "Time's Up!" : "OOM Crash!";
    const t1 = new PIXI.Text({ text: titleStr, style: { fontSize: 28, fontWeight: "bold", fill: titleColor } });
    t1.anchor.set(0.5); t1.position.set(W / 2, H / 2 - 20);
    const t2 = new PIXI.Text({ text: `Completed passes: ${score}`, style: { fontSize: 16, fill: colors.text } });
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
    id: "pipeline",
    name: "Pipeline Pacer",
    ahaLabel: "Pipeline Bubbles",
    ahaText: "In pipeline parallelism, you want many active microbatches to hide bubbles, but too many will cause an Out-Of-Memory error because you must stash their activations for the backward pass.",
    destroy() {
      canvas.removeEventListener("pointerdown", onDown);
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keydown", handleKeydown);
      canvas.removeEventListener("pointerdown", handlePointerdown);
      app.destroy(true, { children: true, texture: true });
    }
  };
}

window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};
window.MLSP.games.pipeline = mountPipeline;
