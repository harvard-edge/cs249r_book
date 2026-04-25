import { mountPixiOnCanvas, burst, floatText } from "/assets/games/runtime.mjs";
import * as P from "/assets/games/vendor/pixi.min.mjs";

window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};
window.MLSP.games.lander = function(canvas, opts) { return mountLander(canvas, opts); };

export async function mountLander(canvas, opts = {}) {
  const { app, stage, width: W, height: H } = await mountPixiOnCanvas(canvas, { bg: 0xfbfbfb });

  const COL = {
    bg: 0xfbfbfb, text: 0x333333,
    ship: 0x4a90c4, thrust: 0xc87b2a,
    pad: 0x3d9e5a, localPad: 0xc87b2a, ground: 0x6f7782,
    crash: 0xc44444
  };

  const state = {
    x: W / 2, y: 40,
    vx: 0, vy: 0,
    angle: 0, // radians
    fuel: 100, // VRAM
    over: false, won: false,
    keys: { up: false, left: false, right: false }
  };

  const gravity = 0.05;
  const thrustPower = 0.15;
  const rotSpeed = 0.08;
  const maxSafeSpeed = 2.0;
  const terrain = createTerrain();

  function rand(min, max) {
    return min + Math.random() * (max - min);
  }

  function createTerrain() {
    const globalX = rand(W * 0.42, W * 0.58);
    const leftX = rand(W * 0.17, W * 0.32);
    const rightX = rand(W * 0.68, W * 0.83);
    return {
      phase: rand(0, Math.PI * 2),
      slope: rand(-8, 8),
      global: { x: globalX, width: rand(86, 112), wellWidth: rand(72, 92), depth: rand(50, 66) },
      locals: [
        { x: leftX, width: rand(70, 92), wellWidth: rand(54, 70), depth: rand(24, 36) },
        { x: rightX, width: rand(70, 92), wellWidth: rand(54, 70), depth: rand(24, 36) }
      ]
    };
  }

  function well(x, center, width, depth) {
    const z = (x - center) / width;
    return depth * Math.exp(-z * z);
  }

  function lossY(x) {
    const normalized = x / W;
    const base = H - 96
      + terrain.slope * (normalized - 0.5)
      + 13 * Math.sin(normalized * Math.PI * 2.1 + terrain.phase)
      + 7 * Math.sin(normalized * Math.PI * 4.7 + terrain.phase * 0.53);
    return base
      + well(x, terrain.global.x, terrain.global.wellWidth, terrain.global.depth)
      + well(x, terrain.locals[0].x, terrain.locals[0].wellWidth, terrain.locals[0].depth)
      + well(x, terrain.locals[1].x, terrain.locals[1].wellWidth, terrain.locals[1].depth);
  }

  function inPad(x, pad) {
    return x >= pad.x - pad.width / 2 && x <= pad.x + pad.width / 2;
  }
  function inGlobalPad(x) { return inPad(x, terrain.global); }
  function inLocalPad(x) {
    return terrain.locals.some(pad => inPad(x, pad));
  }

  const bg = new P.Graphics();
  bg.rect(0, 0, W, H).fill({ color: COL.bg });
  for (let y = 70; y < H - 70; y += 42) {
    bg.moveTo(26, y).lineTo(W - 26, y).stroke({ color: 0xe7eaee, width: 1 });
  }
  for (let x = 40; x < W; x += 70) {
    bg.moveTo(x, 54).lineTo(x, H - 54).stroke({ color: 0xf0f2f4, width: 1 });
  }

  const basinWash = new P.Graphics();
  basinWash.moveTo(0, lossY(0));
  for (let x = 0; x <= W; x += 10) basinWash.lineTo(x, lossY(x));
  basinWash.lineTo(W, H).lineTo(0, H).closePath().fill({ color: 0xeaf2f8 });

  const contourLayer = new P.Graphics();
  for (let offset = 18; offset <= 72; offset += 18) {
    contourLayer.moveTo(0, lossY(0) - offset);
    for (let x = 0; x <= W; x += 10) contourLayer.lineTo(x, lossY(x) - offset);
    contourLayer.stroke({ color: offset === 18 ? 0xd9e7f1 : 0xedf2f5, width: 1 });
  }

  const surface = new P.Graphics();
  surface.moveTo(0, lossY(0));
  for (let x = 0; x <= W; x += 6) surface.lineTo(x, lossY(x));
  surface.stroke({ color: COL.ground, width: 3 });

  const padGlobal = new P.Graphics();
  padGlobal.roundRect(terrain.global.x - terrain.global.width / 2, lossY(terrain.global.x) - 3, terrain.global.width, 6, 3).fill({ color: COL.pad });
  const padLocal1 = new P.Graphics();
  padLocal1.roundRect(terrain.locals[0].x - terrain.locals[0].width / 2, lossY(terrain.locals[0].x) - 3, terrain.locals[0].width, 6, 3).fill({ color: COL.localPad });
  const padLocal2 = new P.Graphics();
  padLocal2.roundRect(terrain.locals[1].x - terrain.locals[1].width / 2, lossY(terrain.locals[1].x) - 3, terrain.locals[1].width, 6, 3).fill({ color: COL.localPad });

  const globalLabel = new P.Text({ text: "global minimum", style: { fill: 0x3d9e5a, fontSize: 11, fontWeight: "bold" }});
  globalLabel.anchor.set(0.5);
  globalLabel.position.set(terrain.global.x, lossY(terrain.global.x) - 18);
  const localLabel = new P.Text({ text: "local minima", style: { fill: 0x9a6620, fontSize: 10 }});
  localLabel.anchor.set(0.5);
  localLabel.position.set(terrain.locals[0].x, lossY(terrain.locals[0].x) - 17);

  stage.addChild(bg, contourLayer, basinWash, surface, padGlobal, padLocal1, padLocal2, globalLabel, localLabel);

  // Ship
  const ship = new P.Graphics();
  ship.moveTo(0, -15).lineTo(10, 10).lineTo(-10, 10).lineTo(0, -15).fill({color: COL.ship});
  stage.addChild(ship);

  // Flame
  const flame = new P.Graphics();
  flame.moveTo(-5, 10).lineTo(0, 25).lineTo(5, 10).fill({color: COL.thrust});
  flame.visible = false;
  ship.addChild(flame);

  const noiseText = new P.Text({ text: "stochastic gradient noise", style: { fill: 0x777777, fontSize: 13 }});
  noiseText.position.set(18, 18);
  stage.addChild(noiseText);

  const handleKeydown = (e) => {
    if (e.key === 'ArrowUp') { e.preventDefault(); state.keys.up = true; }
    if (e.key === 'ArrowLeft') { e.preventDefault(); state.keys.left = true; }
    if (e.key === 'ArrowRight') { e.preventDefault(); state.keys.right = true; }
    if (e.key.toLowerCase() === 'r' && state.over && opts.onRetry) opts.onRetry();
  };
  const handleKeyup = (e) => {
    if (e.key === 'ArrowUp') state.keys.up = false;
    if (e.key === 'ArrowLeft') state.keys.left = false;
    if (e.key === 'ArrowRight') state.keys.right = false;
  };
  window.addEventListener('keydown', handleKeydown);
  window.addEventListener('keyup', handleKeyup);

  app.ticker.add(() => {
    if (state.over) return;

    if (state.keys.left) state.angle -= rotSpeed;
    if (state.keys.right) state.angle += rotSpeed;

    state.vy += gravity;

    if (state.keys.up && state.fuel > 0) {
      state.vx += Math.sin(state.angle) * thrustPower;
      state.vy -= Math.cos(state.angle) * thrustPower;
      state.fuel -= 0.2;
      flame.visible = true;
    } else {
      flame.visible = false;
    }

    state.x += state.vx;
    state.y += state.vy;

    ship.x = state.x;
    ship.y = state.y;
    ship.rotation = state.angle;

    // Check bounds
    if (state.x < 0 || state.x > W) state.over = true; // Off screen
    
    // Landing logic
    const speed = Math.hypot(state.vx, state.vy);
    if (opts.onScoreChange) {
      opts.onScoreChange({ vram: Math.max(0, Math.floor(state.fuel)), speed: Math.floor(speed * 20) });
    }

    // Collision checking
    const groundY = lossY(state.x);

    if (state.y + 10 >= groundY) {
      state.over = true;
      state.y = groundY - 10;
      
      if (inGlobalPad(state.x)) {
        if (speed < maxSafeSpeed && Math.abs(state.angle) < 0.5) {
           state.won = true;
           floatText(stage, state.x, state.y - 30, "CONVERGED!", COL.pad, { size: 24 });
        } else {
           burst(stage, state.x, state.y, COL.crash, 30);
           ship.visible = false;
           floatText(stage, state.x, state.y - 30, "DIVERGED (TOO FAST)", COL.crash, { size: 16 });
        }
      } else if (inLocalPad(state.x)) {
        burst(stage, state.x, state.y, COL.crash, 30);
        ship.visible = false;
        floatText(stage, state.x, state.y - 30, "LOCAL MINIMUM (Suboptimal)", COL.crash, { size: 16 });
      } else {
        burst(stage, state.x, state.y, COL.crash, 30);
        ship.visible = false;
        floatText(stage, state.x, state.y - 30, "GRADIENT EXPLOSION (Crash)", COL.crash, { size: 16 });
      }
    }

    if (state.fuel <= 0 && flame.visible) {
      flame.visible = false;
      floatText(stage, state.x, state.y, "OOM!", COL.crash, {size: 14});
    }

    if (state.over && opts.onGameOver) {
      opts.onGameOver({
         won: state.won
      });
    }
  });

  return {
    id: "lander",
    ahaLabel: "You just experienced",
    ahaText: "Training with a massive batch size gives you a perfectly stable 'thrust' down the loss landscape, but it consumes your VRAM extremely fast. Finding the right balance between Batch Size (fuel burn rate) and Learning Rate (steering angle) is the only way to land in the global minimum without diverging or running out of memory.",
    destroy() {
      window.removeEventListener('keydown', handleKeydown);
      window.removeEventListener('keyup', handleKeyup);
      app.destroy(true, { children: true, texture: true });
    }
  };
}