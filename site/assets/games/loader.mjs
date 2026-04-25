import * as runtime from "./runtime.mjs";

window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};

window.MLSP.games.loader = async function(canvas, callbacks) {
  const { app, stage, width, height, PIXI, onTick, destroy } = await runtime.mountPixiOnCanvas(canvas, { bg: 0x111111 });
  
  const state = {
    score: 0,
    health: 100,
    hunger: 100,
    gameOver: false
  };
  
  const container = new PIXI.Container();
  stage.addChild(container);
  
  // Conveyor Belt line
  const conveyor = new PIXI.Graphics();
  conveyor.moveTo(0, height/2 + 25);
  conveyor.lineTo(width, height/2 + 25);
  conveyor.stroke({ color: 0x444444, width: 2 });
  container.addChild(conveyor);

  // Processing zone
  const zone = new PIXI.Graphics();
  zone.rect(150, height/2 - 50, 300, 100);
  zone.stroke({ color: 0x555555, width: 2, alpha: 0.5 });
  zone.fill({ color: 0x222222, alpha: 0.5 });
  container.addChild(zone);
  
  // GPU Box (Hopper)
  const gpu = new PIXI.Graphics();
  gpu.rect(550, height/2 - 70, 100, 140);
  gpu.fill({ color: 0x004488 });
  gpu.stroke({ color: 0x00aaff, width: 4 });
  container.addChild(gpu);
  
  const gpuText = new PIXI.Text({ text: "GPU", style: { fill: 0xffffff, fontSize: 24, fontWeight: "bold" } });
  gpuText.anchor.set(0.5);
  gpuText.position.set(600, height/2);
  container.addChild(gpuText);

  // Hunger Bar Graphics
  const hungerBg = new PIXI.Graphics();
  hungerBg.rect(550, height/2 + 80, 100, 10);
  hungerBg.fill({ color: 0x333333 });
  container.addChild(hungerBg);

  const hungerBar = new PIXI.Graphics();
  hungerBar.rect(0, 0, 100, 10);
  hungerBar.fill({ color: 0x00ff00 });
  hungerBar.position.set(550, height/2 + 80);
  container.addChild(hungerBar);
  
  function updateHungerBar() {
    hungerBar.scale.x = Math.max(0, state.hunger) / 100;
  }
  updateHungerBar();
  
  const letters = ['J', 'C', 'A', 'T'];
  
  // Object pool to prevent WebGL buffer issues from rapid instantiation
  const blockPool = [];
  for (let i = 0; i < 20; i++) {
    const b = new PIXI.Container();
    const bg = new PIXI.Graphics();
    bg.roundRect(-25, -25, 50, 50, 6);
    bg.fill({ color: 0xc44444 });
    bg.stroke({ color: 0x7f1d1d, width: 2 });
    b.visible = false;
    
    const t = new PIXI.Text({ text: "J", style: { fill: 0xffffff, fontSize: 24, fontWeight: "bold", fontFamily: "Helvetica Neue, Arial" } });
    t.anchor.set(0.5);
    b.addChild(bg);
    b.addChild(t);
    
    container.addChild(b);
    blockPool.push({ sprite: b, bg: bg, text: t, active: false });
  }

  const blocks = [];
  let spawnTimer = 0;
  let spawnInterval = 1500;
  let speed = 0.15;
  
  function spawnBlock() {
    const poolItem = blockPool.find(p => !p.active);
    if (!poolItem) return; // Pool empty
    
    const l = letters[Math.floor(Math.random() * letters.length)];
    poolItem.text.text = l;
    poolItem.bg.clear();
    poolItem.bg.roundRect(-25, -25, 50, 50, 6).fill({ color: 0xc44444 }).stroke({ color: 0x7f1d1d, width: 2 });
    poolItem.sprite.position.set(30, height/2);
    poolItem.sprite.rotation = 0;
    poolItem.sprite.visible = true;
    poolItem.active = true;
    
    const block = { poolItem: poolItem, letter: l, processed: false, x: 30, y: height/2, vx: 0, vy: 0, rotation: 0, bounced: false };
    blocks.push(block);
  }
  
  const handleKey = (e) => {
    if (state.gameOver) return;
    const key = e.key.toUpperCase();
    if (letters.includes(key)) {
      // Find oldest unprocessed block in zone
      const target = blocks.find(b => !b.processed && !b.bounced && b.x > 150 && b.x < 450);
      if (target && target.letter === key) {
        target.processed = true;
        target.poolItem.bg.clear();
        target.poolItem.bg.roundRect(-25, -25, 50, 50, 6).fill({ color: 0x3d9e5a }).stroke({ color: 0x27683b, width: 2 });
        runtime.burst(stage, target.x, target.y, 0x3d9e5a, 10);
      } else {
        runtime.shake(container, 5, 100);
      }
    }
  };
  window.addEventListener('keydown', handleKey);
  
  onTick((dt) => {
    if (state.gameOver) return;
    
    // Drain hunger
    state.hunger -= dt * 0.01;
    if (state.hunger <= 0) {
      state.hunger = 0;
      state.health -= dt * 0.02;
    }
    updateHungerBar();
    
    spawnTimer -= dt;
    if (spawnTimer <= 0) {
      spawnBlock();
      spawnInterval = Math.max(600, spawnInterval - 20); // speed up over time
      spawnTimer = spawnInterval;
    }
    
    speed += dt * 0.000005; // gradually increase speed
    
    for (let i = blocks.length - 1; i >= 0; i--) {
      const b = blocks[i];
      if (b.bounced) {
        b.x += b.vx * (dt / 16.666);
        b.y += b.vy * (dt / 16.666);
        b.rotation += 0.2 * (dt / 16.666);
        b.poolItem.sprite.position.set(b.x, b.y);
        b.poolItem.sprite.rotation = b.rotation;
        
        if (b.y > height + 50 || b.x < -50 || b.x > width + 50) {
          b.poolItem.active = false;
          b.poolItem.sprite.visible = false;
          blocks.splice(i, 1);
        }
      } else {
        b.x += speed * dt;
        b.poolItem.sprite.position.x = b.x;
        
        if (b.x > 550) {
          if (!b.processed) {
            // RAW BLOCK bounces off
            state.health -= 15;
            runtime.flash(stage, 0xff0000, 200, 0.3);
            runtime.shake(container, 10, 200);
            b.bounced = true;
            b.vx = -2;
            b.vy = 2;
            b.rotation = 0;
          } else {
            // PROCESSED TENSOR hits GPU
            state.score += 10;
            state.hunger = Math.min(100, state.hunger + 25);
            runtime.floatText(stage, 600, height/2 - 90, "+10", 0x00ff00);
            runtime.burst(stage, 600, height/2, 0x00ff00, 20);
            b.poolItem.active = false;
            b.poolItem.sprite.visible = false;
            blocks.splice(i, 1);
          }
          callbacks.onScoreChange(state);
          
          if (state.health <= 0 && !state.gameOver) {
            state.gameOver = true;
            endGame();
          }
        }
      }
    }
  });
  
  function endGame() {
    const go = new PIXI.Text({ text: "GAME OVER\nPress R to Retry", style: { fill: 0xffffff, fontSize: 48, align: 'center' } });
    go.anchor.set(0.5);
    go.position.set(width/2, height/2);
    stage.addChild(go);
    callbacks.onGameOver({ score: state.score });
  }
  
  return {
    ahaLabel: "Data Loading",
    ahaText: "If the CPU can't decode and prep data fast enough, the GPU sits idle.",
    ahaLink: { href: "/", label: "Read Vol I: Data Engineering" },
    destroy: () => {
      window.removeEventListener('keydown', handleKey);
      destroy();
    }
  };
};