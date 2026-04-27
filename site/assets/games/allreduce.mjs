import { mountPixiOnCanvas, burst, flash, floatText, shake, tween, mountReadyOverlay } from "./runtime.mjs";
import * as P from "./vendor/pixi.min.mjs";

window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};
window.MLSP.games.allreduce = function(canvas, opts) { return mountAllreduce(canvas, opts); };

export async function mountAllreduce(canvas, opts = {}) {
  const { app, stage, width: W, height: H, onTick } = await mountPixiOnCanvas(canvas, { bg: 0x11151c });

  const COL = {
    bg: 0x11151c,
    gpu: 0x2a3b4c,
    gpuActive: 0x4a90c4,
    wire: 0x334455,
    chunk: 0x3d9e5a,
    perfect: 0x3d9e5a,
    miss: 0xc44444,
    text: 0xeeeeee
  };

  const R = 120; // radius of the ring
  const cx = W / 2;
  const cy = H / 2;
  const BPM = 60;
  const beatInterval = 60000 / BPM;
  const tolerance = 150; // ms

  const state = {
    combo: 0,
    score: 0,
    time: 0,
    lastBeatTime: 0,
    over: false,
    started: false
  };

  const gpus = [];
  const wires = new P.Graphics();
  stage.addChild(wires);

  const gameLayer = new P.Container();
  stage.addChild(gameLayer);

  // Draw circular wires
  wires.circle(cx, cy, R).stroke({ width: 4, color: COL.wire });

  // Metronome ring
  const metronome = new P.Graphics();
  metronome.circle(0, 0, R).stroke({ width: 2, color: 0xffffff });
  metronome.position.set(cx, cy);
  gameLayer.addChild(metronome);

  for (let i = 0; i < 4; i++) {
    const angle = (i * Math.PI) / 2 - Math.PI / 4; // 45, 135, 225, 315 deg
    const x = cx + R * Math.cos(angle);
    const y = cy + R * Math.sin(angle);

    const gpuContainer = new P.Container();
    gpuContainer.position.set(x, y);

    const box = new P.Graphics();
    box.roundRect(-25, -25, 50, 50, 8).fill({ color: COL.gpu });
    
    const label = new P.Text({
      text: (i + 1).toString(),
      style: { fill: 0xffffff, fontSize: 20, fontWeight: "bold" }
    });
    label.anchor.set(0.5);

    gpuContainer.addChild(box, label);
    gameLayer.addChild(gpuContainer);

    gpuContainer.eventMode = 'static';
    gpuContainer.cursor = 'pointer';
    gpuContainer.on('pointerdown', () => handleInput(i));

    gpus.push({ x, y, box, angle, container: gpuContainer });
  }

  function spawnChunk(fromIdx) {
    const toIdx = (fromIdx + 1) % 4;
    const chunk = new P.Graphics();
    chunk.roundRect(-10, -10, 20, 20, 4).fill({ color: COL.chunk });
    gameLayer.addChild(chunk);
    
    // Tween along arc or just straight line? Let's do straight line for simplicity, 
    // or manually calculate arc in ticker. Straight line is fast.
    const startObj = gpus[fromIdx];
    const endObj = gpus[toIdx];
    chunk.position.set(startObj.x, startObj.y);

    const duration = beatInterval * 0.8;
    const cancelTween = tween(
      [chunk.position, chunk.position],
      ["x", "y"],
      [startObj.x, startObj.y],
      [endObj.x, endObj.y],
      duration,
      "linear"
    );
    
    setTimeout(() => {
      chunk.destroy();
      burst(gameLayer, endObj.x, endObj.y, COL.chunk, 10, { speed: 1.5 });
    }, duration);
  }

  function handleInput(idx) {
    if (!state.started || state.over) return;
    
    // Check timing
    const nextBeat = state.lastBeatTime + beatInterval;
    const diff = Math.min(Math.abs(state.time - state.lastBeatTime), Math.abs(state.time - nextBeat));

    const box = gpus[idx].box;
    box.tint = 0xffffff;
    setTimeout(() => { if (!box.destroyed) box.tint = 0xffffff; }, 100);

    if (diff <= tolerance) {
      // Perfect
      state.combo++;
      state.score += 10 * state.combo;
      flash(stage, COL.perfect, 150, 0.2);
      burst(gameLayer, gpus[idx].x, gpus[idx].y, COL.perfect, 12);
      spawnChunk(idx);
      if (opts.onScoreChange) opts.onScoreChange({ score: state.score, combo: state.combo });
    } else {
      // Miss
      state.combo = 0;
      flash(stage, COL.miss, 200, 0.4);
      shake(gameLayer, 10, 200);
      floatText(gameLayer, gpus[idx].x, gpus[idx].y - 30, "COLLISION", COL.miss, { size: 18 });
      if (opts.onScoreChange) opts.onScoreChange({ score: state.score, combo: state.combo });
    }
  }

  window.addEventListener('keydown', (e) => {
    if (e.key >= '1' && e.key <= '4') {
      handleInput(parseInt(e.key) - 1);
    }
  });

  // Pre-game READY overlay
  mountReadyOverlay(stage, {
    width: W, height: H,
    title: "ALL-REDUCE RHYTHM",
    goal: "Tap each GPU on the beat. Keep gradients flowing.",
    controls: "1 2 3 4  fire to GPU · TAP a GPU · R  retry",
    onLaunch: () => { state.started = true; state.lastBeatTime = state.time; }
  });

  onTick((dt) => {
    if (!state.started || state.over) return;
    state.time += dt;

    if (state.time >= state.lastBeatTime + beatInterval) {
      state.lastBeatTime += beatInterval;
    }

    // Update metronome visual
    const progress = (state.time - state.lastBeatTime) / beatInterval;
    const scale = Math.max(0.01, 1 - progress);
    metronome.scale.set(scale);
    metronome.alpha = 0.5 * (1 - progress);
  });

  return {
    id: "allreduce",
    ahaLabel: "You just experienced",
    ahaText: "Ring All-Reduce synchronizes gradients across GPUs by passing chunks in a circle. Perfect timing (synchronous steps) ensures maximum bandwidth utilization. When GPUs fall out of sync, collisions and pipeline stalls occur, cratering your training throughput."
  };
}