import { mountPixiOnCanvas, burst, floatText } from "/assets/games/runtime.mjs";
import * as P from "/assets/games/vendor/pixi.min.mjs";

window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};
window.MLSP.games.lander = function(canvas, opts) { return mountLander(canvas, opts); };

export async function mountLander(canvas, opts = {}) {
  const { app, stage, width: W, height: H } = await mountPixiOnCanvas(canvas, { bg: 0xfafafa });

  const COL = {
    bg: 0xfafafa, text: 0x333333,
    ship: 0x4a90c4, thrust: 0xc87b2a,
    pad: 0x3d9e5a, ground: 0x888888,
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

  // Draw ground
  const ground = new P.Graphics();
  ground.moveTo(0, H - 40)
        .lineTo(W/4 - 40, H - 40).lineTo(W/4 - 40, H - 20).lineTo(W/4 + 40, H - 20).lineTo(W/4 + 40, H - 40) // Local Min 1
        .lineTo(W/2 - 50, H - 40).lineTo(W/2 - 50, H - 10).lineTo(W/2 + 50, H - 10).lineTo(W/2 + 50, H - 40) // Global Min (Deep)
        .lineTo(3*W/4 - 40, H - 40).lineTo(3*W/4 - 40, H - 20).lineTo(3*W/4 + 40, H - 20).lineTo(3*W/4 + 40, H - 40) // Local Min 2
        .lineTo(W, H - 40).lineTo(W, H).lineTo(0, H).fill({color: COL.ground});
  
  const padGlobal = new P.Graphics();
  padGlobal.rect(W/2 - 50, H - 10, 100, 6).fill({color: COL.pad}); // Global Min
  
  const padLocal1 = new P.Graphics();
  padLocal1.rect(W/4 - 40, H - 20, 80, 6).fill({color: 0xc87b2a}); // Local Min 1

  const padLocal2 = new P.Graphics();
  padLocal2.rect(3*W/4 - 40, H - 20, 80, 6).fill({color: 0xc87b2a}); // Local Min 2

  stage.addChild(ground, padGlobal, padLocal1, padLocal2);

  // Ship
  const ship = new P.Graphics();
  ship.moveTo(0, -15).lineTo(10, 10).lineTo(-10, 10).lineTo(0, -15).fill({color: COL.ship});
  stage.addChild(ship);

  // Flame
  const flame = new P.Graphics();
  flame.moveTo(-5, 10).lineTo(0, 25).lineTo(5, 10).fill({color: COL.thrust});
  flame.visible = false;
  ship.addChild(flame);

  const noiseText = new P.Text({ text: "[Stochastic Noise Active]", style: { fill: 0x888888, fontSize: 14 }});
  noiseText.position.set(10, 10);
  stage.addChild(noiseText);

  window.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowUp') { e.preventDefault(); state.keys.up = true; }
    if (e.key === 'ArrowLeft') { e.preventDefault(); state.keys.left = true; }
    if (e.key === 'ArrowRight') { e.preventDefault(); state.keys.right = true; }
    if (e.key.toLowerCase() === 'r' && state.over && opts.onRetry) opts.onRetry();
  });
  window.addEventListener('keyup', (e) => {
    if (e.key === 'ArrowUp') state.keys.up = false;
    if (e.key === 'ArrowLeft') state.keys.left = false;
    if (e.key === 'ArrowRight') state.keys.right = false;
  });

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
    let groundY = H - 40;
    if (state.x >= W/2 - 50 && state.x <= W/2 + 50) groundY = H - 10;
    else if (state.x >= W/4 - 40 && state.x <= W/4 + 40) groundY = H - 20;
    else if (state.x >= 3*W/4 - 40 && state.x <= 3*W/4 + 40) groundY = H - 20;

    if (state.y + 10 >= groundY) {
      state.over = true;
      state.y = groundY - 10;
      
      if (groundY === H - 10) {
        if (speed < maxSafeSpeed && Math.abs(state.angle) < 0.5) {
           state.won = true;
           floatText(stage, state.x, state.y - 30, "CONVERGED!", COL.pad, { size: 24 });
        } else {
           burst(stage, state.x, state.y, COL.crash, 30);
           ship.visible = false;
           floatText(stage, state.x, state.y - 30, "DIVERGED (TOO FAST)", COL.crash, { size: 16 });
        }
      } else if (groundY === H - 20) {
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
    ahaText: "Training with a massive batch size gives you a perfectly stable 'thrust' down the loss landscape, but it consumes your VRAM extremely fast. Finding the right balance between Batch Size (fuel burn rate) and Learning Rate (steering angle) is the only way to land in the global minimum without diverging or running out of memory."
  };
}