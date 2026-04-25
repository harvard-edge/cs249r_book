import * as runtime from "./runtime.mjs";

window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};

window.MLSP.games.roofline = async function(canvas, callbacks) {
  const { app, stage, width, height, PIXI, onTick, destroy } = await runtime.mountPixiOnCanvas(canvas, { bg: 0x050510 });
  
  const state = {
    score: 0,
    health: 100,
    gameOver: false
  };
  
  const container = new PIXI.Container();
  stage.addChild(container);
  
  // Roofline coords
  const roofX1 = 50, roofY1 = 400; // Origin
  const roofX2 = 250, roofY2 = 100; // Ridge
  const roofX3 = 680, roofY3 = 100; // Flat
  
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
  
  const player = new PIXI.Graphics();
  player.circle(0, 0, 8);
  player.fill({ color: 0xff00ff });
  container.addChild(player);
  
  let px = 100;
  let py = 350;
  
  const keys = { ArrowUp: false, ArrowDown: false };
  const downHandler = (e) => { if(keys.hasOwnProperty(e.code)) { keys[e.code] = true; e.preventDefault(); } };
  const upHandler = (e) => { if(keys.hasOwnProperty(e.code)) { keys[e.code] = false; e.preventDefault(); } };
  window.addEventListener('keydown', downHandler, {passive: false});
  window.addEventListener('keyup', upHandler, {passive: false});
  
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
  
  onTick((dt) => {
    if (state.gameOver) return;
    
    if (keys.ArrowUp) py -= dt * 0.25;
    if (keys.ArrowDown) py += dt * 0.25;
    
    if (py > 400) py = 400;
    if (py < 50) py = 50;
    
    player.position.set(px, py);
    
    let roofY = 400;
    if (px < roofX2) {
      const t = (px - roofX1) / (roofX2 - roofX1);
      roofY = roofY1 + t * (roofY2 - roofY1);
    } else {
      roofY = roofY2;
    }
    
    if (py < roofY) { 
      state.health -= 0.8 * dt;
      py += dt * 0.5;
      runtime.shake(container, 5, 50);
      player.tint = 0xff0000;
      if (Math.random() < 0.05) {
        runtime.floatText(stage, px, py - 20, "COMPUTE ROOF!", 0xff0000, { size: 16 });
      }
    } else {
      player.tint = 0xffffff;
      state.score += (400 - py) * dt * 0.01;
      
      if (Math.random() < 0.005 && py > 250) {
        runtime.floatText(stage, px, py + 20, "Fly higher for more points!", 0xffff00, { size: 14 });
      }
    }
    
    wallTimer -= dt;
    if (wallTimer <= 0) {
      spawnWall();
      wallTimer = 1000 + Math.random() * 1500;
    }
    
    for (let i = walls.length - 1; i >= 0; i--) {
      const w = walls[i];
      w.x -= dt * 0.2;
      w.sprite.position.x = w.x;
      
      if (px > w.x && px < w.x + w.w && py > w.y && py < w.y + w.h) {
        state.health -= 0.5 * dt;
        runtime.shake(container, 8, 50);
      }
      
      if (w.x < -50) {
        w.sprite.destroy();
        walls.splice(i, 1);
      }
    }
    
    callbacks.onScoreChange({ score: Math.floor(state.score), health: Math.floor(state.health) });
    
    if (state.health <= 0) {
      state.gameOver = true;
      const go = new PIXI.Text({ text: "CRASHED\nPress R to Retry", style: { fill: 0xffffff, fontSize: 48, align: 'center' } });
      go.anchor.set(0.5);
      go.position.set(width/2, height/2);
      stage.addChild(go);
      callbacks.onGameOver({ score: Math.floor(state.score) });
    }
  });
  
  return {
    ahaLabel: "Roofline Model",
    ahaText: "You can't exceed memory bandwidth (the slope) or peak compute (the flat roof).",
    ahaLink: { href: "/", label: "Read Vol I: Hardware Acceleration" },
    destroy: () => {
      window.removeEventListener('keydown', downHandler);
      window.removeEventListener('keyup', upHandler);
      destroy();
    }
  };
};