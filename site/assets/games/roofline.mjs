import * as runtime from "./runtime.mjs";

window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};

window.MLSP.games.roofline = async function(canvas, callbacks) {
  const { app, stage, width, height, PIXI, onTick, destroy } = await runtime.mountPixiOnCanvas(canvas, { bg: 0x050510 });
  
  const state = {
    score: 0,
    health: 100,
    gameOver: false,
    started: false
  };
  
  const container = new PIXI.Container();
  stage.addChild(container);
  
  // Horizontal position is operational intensity; vertical position is performance.
  const roofX1 = 50, roofY1 = 400;
  const roofX2 = 250, roofY2 = 100;
  const roofX3 = 680, roofY3 = 100;
  
  const roof = new PIXI.Graphics();
  roof.moveTo(roofX1, roofY1);
  roof.lineTo(roofX2, roofY2);
  roof.lineTo(roofX3, roofY3);
  roof.stroke({ color: 0x00ffff, width: 4 });
  
  try {
    const filters = await runtime.getFilters();
    if (filters && filters.GlowFilter) {
      roof.filters = [new filters.GlowFilter({ distance: 15, outerStrength: 2, color: 0x00ffff })];
    }
  } catch(e) {}
  
  container.addChild(roof);

  const axisStyle = { fill: 0xa8b8c8, fontSize: 13, fontWeight: "bold" };
  const performanceLabel = new PIXI.Text({ text: "PERFORMANCE ↑", style: axisStyle });
  performanceLabel.position.set(20, 15);
  container.addChild(performanceLabel);
  const intensityLabel = new PIXI.Text({ text: "OPERATIONAL INTENSITY →", style: axisStyle });
  intensityLabel.position.set(430, 430);
  container.addChild(intensityLabel);
  const phaseLabel = new PIXI.Text({ text: "MEMORY BOUND", style: { fill: 0x80e5ff, fontSize: 16, fontWeight: "bold" } });
  phaseLabel.position.set(330, 18);
  container.addChild(phaseLabel);
  
  const player = new PIXI.Graphics();
  player.circle(0, 0, 8);
  player.fill({ color: 0xff00ff });
  container.addChild(player);
  
  let px = 75;
  let py = 385;
  
  const keys = { ArrowUp: false, ArrowDown: false };
  const downHandler = (e) => { if(keys.hasOwnProperty(e.code)) { keys[e.code] = true; e.preventDefault(); } };
  const upHandler = (e) => { if(keys.hasOwnProperty(e.code)) { keys[e.code] = false; e.preventDefault(); } };
  window.addEventListener('keydown', downHandler, {passive: false});
  window.addEventListener('keyup', upHandler, {passive: false});
  const steer = (direction, pressed) => { keys[direction === 'up' ? 'ArrowUp' : 'ArrowDown'] = pressed; };
  
  const walls = [];
  let wallTimer = 1000;
  
  function spawnWall() {
    const w = new PIXI.Graphics();
    const h = 50 + Math.random() * 120;
    const isTop = Math.random() > 0.5;
    const yPos = isTop ? 50 : 400 - h;
    w.rect(0, 0, 20, h);
    w.fill({ color: 0xff5500, alpha: 0.7 });
    w.position.set(680, yPos);
    container.addChild(w);
    walls.push({ sprite: w, x: 680, y: yPos, w: 20, h: h });
  }
  
  const trail = new PIXI.Graphics();
  container.addChildAt(trail, 0);
  const history = [];
  
  // Pre-game READY overlay
  runtime.mountReadyOverlay(stage, {
    width: width, height: height,
    title: "ROOFLINE RIDER",
    goal: "Ride from the memory limit to the compute limit.",
    controls: "↑ ↓ or hold the buttons · chase the cyan roof · dodge orange stalls",
    onLaunch: () => { state.started = true; }
  });

  onTick((dt) => {
    if (!state.started) return;
    if (state.gameOver) return;

    px = Math.min(630, px + dt * 0.021);
    phaseLabel.text = px < roofX2 ? "MEMORY BOUND" : "COMPUTE BOUND";
    if (keys.ArrowUp) py -= dt * 0.25;
    if (keys.ArrowDown) py += dt * 0.25;
    
    if (py > 400) py = 400;
    if (py < 50) py = 50;
    
    let roofY = 400;
    if (px < roofX2) {
      const t = (px - roofX1) / (roofX2 - roofX1);
      roofY = roofY1 + t * (roofY2 - roofY1);
    } else {
      roofY = roofY2;
    }
    
    if (py < roofY) { 
      state.health = Math.max(0, state.health - 0.12 * dt);
      py += dt * 0.5;
      runtime.shake(container, 5, 50);
      player.tint = 0xff0000;
      if (Math.random() < 0.05) {
        runtime.floatText(stage, px, py - 20, px < roofX2 ? "MEMORY LIMIT!" : "COMPUTE LIMIT!", 0xff0000, { size: 16 });
      }
    } else {
      player.tint = 0xffffff;
      // Only performance close to the current hardware limit earns points.
      state.score += Math.max(0, 100 - (py - roofY)) * dt * 0.001;
      
      if (Math.random() < 0.005 && py > 250) {
        runtime.floatText(stage, px, py + 20, "Chase the roof for more points!", 0xffff00, { size: 14 });
      }
    }
    player.position.set(px, py);
    
    wallTimer -= dt;
    if (wallTimer <= 0 && px < 440) {
      spawnWall();
      wallTimer = 1000 + Math.random() * 1500;
    }
    
    for (let i = walls.length - 1; i >= 0; i--) {
      const w = walls[i];
      w.x -= dt * 0.2;
      w.sprite.position.x = w.x;
      
      if (px > w.x && px < w.x + w.w && py > w.y && py < w.y + w.h) {
        state.health = Math.max(0, state.health - 0.18 * dt);
        runtime.shake(container, 8, 50);
      }
      
      if (w.x < -50) {
        w.sprite.destroy();
        walls.splice(i, 1);
      }
    }
    
    callbacks.onScoreChange({ score: Math.floor(state.score), health: Math.floor(state.health), phase: px < roofX2 ? "memory bound" : "compute bound" });
    
    if (state.health <= 0 || px >= 630) {
      state.gameOver = true;
      const completed = state.health > 0;
      const resultText = !completed ? "CRASHED" : state.score >= 900 ? "ROOFLINE RIDER!" : "SAFE, BUT SLOW!";
      const panel = new PIXI.Graphics();
      panel.roundRect(width / 2 - 245, height / 2 - 72, 490, 144, 14)
        .fill({ color: 0x101827, alpha: 0.96 })
        .stroke({ color: 0x00ffff, width: 2 });
      stage.addChild(panel);
      const go = new PIXI.Text({ text: resultText + "\nTap retry or press R", style: { fill: 0xffffff, fontSize: 36, fontWeight: 'bold', align: 'center' } });
      go.anchor.set(0.5);
      go.position.set(width/2, height/2);
      stage.addChild(go);
      callbacks.onGameOver({ score: Math.floor(state.score), completed });
    }
  });
  
  return {
    steer,
    ahaLabel: "Roofline Model",
    ahaText: "The sloped roof is the memory bandwidth limit. Past the knee, the flat roof is peak compute. Higher intensity helps only until you reach that plateau.",
    ahaLink: { href: "/vol1/hw_acceleration/hw_acceleration.html", label: "Read Vol I: Hardware Acceleration" },
    destroy: () => {
      window.removeEventListener('keydown', downHandler);
      window.removeEventListener('keyup', upHandler);
      destroy();
    }
  };
};
