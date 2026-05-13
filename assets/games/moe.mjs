import { mountPixiOnCanvas, burst, flash, shake, tween, mountReadyOverlay } from "./runtime.mjs";
import * as PIXI from "./vendor/pixi.min.mjs";

export async function mountMoe(canvas, opts = {}) {
  const { app, stage, width: W, height: H } = await mountPixiOnCanvas(canvas, { bg: 0xf8f9fa });

  let score = 0;
  let over = false;
  let started = false;
  let tokens = [];
  let nextTokenId = 0;
  let timeElapsed = 0;
  let spawnTimer = 0;

  const bgLayer = new PIXI.Container();
  const tokenLayer = new PIXI.Container();
  const hudLayer = new PIXI.Container();
  const overlayLayer = new PIXI.Container();
  stage.addChild(bgLayer, tokenLayer, hudLayer, overlayLayer);

  const EXPERTS = [
    { id: 1, color: 0xc44444, key: "1" }, // Red
    { id: 2, color: 0x4a90c4, key: "2" }, // Blue
    { id: 3, color: 0x3d9e5a, key: "3" }, // Green
    { id: 4, color: 0xc87b2a, key: "4" }  // Orange
  ];
  
  const expertWidth = 60;
  const spacing = 15;
  const totalWidth = 4 * expertWidth + 3 * spacing;
  const startX = W / 2 - totalWidth / 2;
  const expertY = H - 60;

  EXPERTS.forEach((exp, i) => {
    const x = startX + i * (expertWidth + spacing);
    exp.x = x + expertWidth / 2;
    const box = new PIXI.Graphics();
    box.roundRect(x, expertY, expertWidth, 40, 8).fill(exp.color);
    box.eventMode = "static";
    box.on("pointerdown", () => routeToken(i));
    bgLayer.addChild(box);
    
    const text = new PIXI.Text({ text: exp.key, style: { fontSize: 16, fontWeight: "bold", fill: 0xffffff } });
    text.anchor.set(0.5);
    text.position.set(x + expertWidth / 2, expertY + 20);
    bgLayer.addChild(text);
  });

  const title = new PIXI.Text({ text: "MoE Router", style: { fontSize: 16, fontWeight: "bold", fill: 0x333333 } });
  title.position.set(20, 15);
  const scoreText = new PIXI.Text({ text: "Score: 0", style: { fontSize: 14, fill: 0x333333 } });
  scoreText.position.set(20, 40);
  const hintText = new PIXI.Text({ text: "Press 1-4 or tap experts to route tokens", style: { fontSize: 12, fill: 0x6c757d, fontStyle: "italic" } });
  hintText.anchor.set(0.5); hintText.position.set(W / 2, 30);
  hudLayer.addChild(title, scoreText, hintText);

  function spawnToken() {
    if (over) return;
    const typeIdx = Math.floor(Math.random() * EXPERTS.length);
    const exp = EXPERTS[typeIdx];
    const token = {
      id: nextTokenId++,
      typeIdx,
      color: exp.color,
      y: 60,
      sprite: new PIXI.Graphics()
    };
    token.sprite.circle(0, 0, 12).fill(exp.color).stroke({ width: 2, color: 0xffffff });
    token.sprite.position.set(W / 2, token.y);
    tokenLayer.addChild(token.sprite);
    tokens.push(token);
  }

  function routeToken(expertIndex) {
    if (!started || over || tokens.length === 0) return;
    // Find lowest token
    const token = tokens[0];
    tokens.shift();
    
    const exp = EXPERTS[expertIndex];
    // Zoom to expert
    tween(token.sprite, ["position.x", "position.y"], 
          [token.sprite.position.x, token.sprite.position.y], 
          [exp.x, expertY], 150, "linear");
          
    setTimeout(() => {
      if (over) return;
      token.sprite.destroy();
      if (token.typeIdx === expertIndex) {
        burst(tokenLayer, exp.x, expertY, exp.color, 6);
        score++;
        scoreText.text = "Score: " + score;
        if (opts.onScoreChange) opts.onScoreChange({ score, timeLeft: 30000 });
      } else {
        // Wrong route!
        over = true;
        flash(stage, 0xc44444, 400);
        shake(stage, 15, 400);
        drawGameOver();
      }
    }, 150);
  }

  const onKey = (e) => {
    if (over) return;
    if (e.key === "1") routeToken(0);
    if (e.key === "2") routeToken(1);
    if (e.key === "3") routeToken(2);
    if (e.key === "4") routeToken(3);
  };
  window.addEventListener("keydown", onKey);

  // Pre-game READY overlay
  mountReadyOverlay(stage, {
    width: W, height: H,
    title: "MoE ROUTER",
    goal: "Route each colored token to the matching expert.",
    controls: "1 2 3 4  route to expert · TAP an expert · R  retry",
    onLaunch: () => { started = true; }
  });

  app.ticker.add((ticker) => {
    const dt = ticker.deltaMS;
    if (!started) return;
    if (over) return;

    timeElapsed += dt / 1000;
    spawnTimer -= dt;
    
    const spawnInterval = Math.max(400, 1500 - timeElapsed * 30);
    if (spawnTimer <= 0) {
      spawnToken();
      spawnTimer = spawnInterval;
    }

    const fallSpeed = 0.05 + timeElapsed * 0.002;
    for (let i = 0; i < tokens.length; i++) {
      const t = tokens[i];
      t.y += dt * fallSpeed;
      t.sprite.position.y = t.y;
      
      if (t.y > expertY - 20) {
        // Missed token!
        over = true;
        flash(stage, 0xc44444, 400);
        shake(stage, 15, 400);
        drawGameOver();
      }
    }
  });

  function drawGameOver() {
    if (opts.onGameOver) opts.onGameOver({ score, timeLeft: 30000 });
    const bg = new PIXI.Graphics();
    bg.rect(0, 0, W, H).fill({ color: 0xffffff, alpha: 0.9 });
    const t1 = new PIXI.Text({ text: "Routing Error!", style: { fontSize: 28, fontWeight: "bold", fill: 0xc44444 } });
    t1.anchor.set(0.5); t1.position.set(W / 2, H / 2 - 20);
    const t2 = new PIXI.Text({ text: `Routed: ${score} tokens`, style: { fontSize: 16, fill: 0x333333 } });
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
    id: "moe",
    name: "MoE Router",
    ahaLabel: "Mixture of Experts",
    ahaText: "MoE models route different tokens to specialized 'expert' subnetworks. This increases capacity without proportionally increasing compute cost per token.",
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
window.MLSP.games.moe = mountMoe;
