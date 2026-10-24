import { mountPixiOnCanvas, burst, flash, floatText, shake, tween, mountReadyOverlay } from "./runtime.mjs";
import * as P from "./vendor/pixi.min.mjs";

window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};
window.MLSP.games.allreduce = function(canvas, opts) { return mountAllreduce(canvas, opts); };

export async function mountAllreduce(canvas, opts = {}) {
  const { stage, width: W, height: H, onTick, destroy } = await mountPixiOnCanvas(canvas, { bg: 0x11151c });

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
  const runTime = 30000;

  const state = {
    combo: 0,
    score: 0,
    time: 0,
    lastBeatTime: 0,
    lastResolvedBeat: 0,
    misses: 0,
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

  const targetText = new P.Text({
    text: "NEXT GPU 1",
    style: { fill: COL.text, fontSize: 19, fontWeight: "700" }
  });
  targetText.anchor.set(0.5);
  targetText.position.set(cx, cy);
  gameLayer.addChild(targetText);
  const timeText = new P.Text({
    text: "30s",
    style: { fill: 0xb8cbd8, fontSize: 18, fontWeight: "700" }
  });
  timeText.anchor.set(1, 0);
  timeText.position.set(W - 26, 20);
  stage.addChild(timeText);

  for (let i = 0; i < 4; i++) {
    const angle = (i * Math.PI) / 2 - Math.PI / 4; // 45, 135, 225, 315 deg
    const x = cx + R * Math.cos(angle);
    const y = cy + R * Math.sin(angle);

    const gpuContainer = new P.Container();
    gpuContainer.position.set(x, y);

    const box = new P.Graphics();
    // Fill white and tint down to the base color so a white tint can flash the box on tap
    box.roundRect(-25, -25, 50, 50, 8).fill({ color: 0xffffff });
    box.tint = COL.gpu;
    
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
      chunk,
      ["position.x", "position.y"],
      [startObj.x, startObj.y],
      [endObj.x, endObj.y],
      duration,
      "linear"
    );

    setTimeout(() => {
      cancelTween();
      chunk.destroy();
      burst(gameLayer, endObj.x, endObj.y, COL.chunk, 10, { speed: 1.5 });
    }, duration);
  }

  function handleInput(idx) {
    if (!state.started || state.over) return;

    const beat = Math.round(state.time / beatInterval);
    const diff = Math.abs(state.time - beat * beatInterval);
    const expected = (beat - 1) % 4;

    const box = gpus[idx].box;
    box.tint = 0xffffff;
    setTimeout(() => { if (!box.destroyed) box.tint = COL.gpu; }, 100);

    if (beat >= 1 && diff <= tolerance && beat > state.lastResolvedBeat && idx === expected) {
      state.lastResolvedBeat = beat;
      state.combo++;
      state.score += 10 * state.combo;
      flash(stage, COL.perfect, 150, 0.2);
      burst(gameLayer, gpus[idx].x, gpus[idx].y, COL.perfect, 12);
      spawnChunk(idx);
      if (opts.onScoreChange) opts.onScoreChange({ score: state.score, combo: state.combo, timeLeft: Math.max(0, runTime - state.time) });
    } else {
      state.combo = 0;
      state.misses++;
      flash(stage, COL.miss, 200, 0.4);
      shake(gameLayer, 10, 200);
      floatText(gameLayer, gpus[idx].x, gpus[idx].y - 30, idx !== expected && beat >= 1 && diff <= tolerance ? "WRONG GPU" : "OFF BEAT", COL.miss, { size: 18 });
      if (opts.onScoreChange) opts.onScoreChange({ score: state.score, combo: state.combo, timeLeft: Math.max(0, runTime - state.time) });
    }
  }

  function handleKeydown(e) {
    if (e.key >= '1' && e.key <= '4') {
      handleInput(parseInt(e.key) - 1);
    } else if (state.over && e.key.toLowerCase() === 'r') {
      opts.onRetry?.();
    }
  }
  window.addEventListener('keydown', handleKeydown);

  // Pre-game READY overlay
  mountReadyOverlay(stage, {
    width: W, height: H,
    title: "ALL-REDUCE RHYTHM",
    goal: "Follow GPUs 1 → 4 on the beat for 30 seconds.",
    controls: "1 2 3 4  fire to GPU · TAP a GPU · R  retry",
    onLaunch: () => { state.started = true; state.lastBeatTime = state.time; }
  });

  onTick((dt) => {
    if (!state.started || state.over) return;
    state.time = Math.min(runTime, state.time + Math.min(dt, 100));
    timeText.text = Math.ceil((runTime - state.time) / 1000) + "s";
    const upcomingBeat = Math.floor(state.time / beatInterval) + 1;
    targetText.text = "NEXT GPU " + ((upcomingBeat - 1) % 4 + 1);
    opts.onScoreChange?.({ score: state.score, combo: state.combo, timeLeft: runTime - state.time });

    if (state.time >= runTime) { finish(); return; }

    const expiredBeat = Math.floor((state.time - tolerance) / beatInterval);
    if (expiredBeat >= 1 && expiredBeat > state.lastResolvedBeat) {
      state.lastResolvedBeat = expiredBeat;
      state.combo = 0;
      state.misses++;
      floatText(gameLayer, cx, cy + 32, "MISSED SLOT", COL.miss, { size: 17 });
    }

    if (state.time >= state.lastBeatTime + beatInterval) {
      state.lastBeatTime += beatInterval;
    }

    // Update metronome visual
    const progress = (state.time - state.lastBeatTime) / beatInterval;
    const scale = Math.max(0.01, 1 - progress);
    metronome.scale.set(scale);
    metronome.alpha = 0.5 * (1 - progress);
  });

  function finish() {
    if (state.over) return;
    state.over = true;
    const shade = new P.Graphics();
    shade.rect(0, 0, W, H).fill(COL.bg);
    const title = new P.Text({
      text: "ROUND COMPLETE",
      style: { fill: COL.perfect, fontSize: 32, fontWeight: "700" }
    });
    title.anchor.set(0.5);
    title.position.set(cx, cy - 42);
    const summary = new P.Text({
      text: `Score ${state.score} · Misses ${state.misses}`,
      style: { fill: COL.text, fontSize: 17 }
    });
    summary.anchor.set(0.5);
    summary.position.set(cx, cy + 7);
    const retry = new P.Text({
      text: "Press R to try again",
      style: { fill: 0xffd6a8, fontSize: 16 }
    });
    retry.anchor.set(0.5);
    retry.position.set(cx, cy + 47);
    stage.addChild(shade, title, summary, retry);
    opts.onGameOver?.({ score: state.score, misses: state.misses });
  }

  return {
    id: "allreduce",
    ahaLabel: "You just experienced",
    ahaText: "Ring all-reduce passes gradient chunks between GPUs in ordered steps. A delayed participant makes peers wait and leaves communication slots idle. The rhythm game is a timing analogy, not a packet-level simulation.",
    destroy() {
      window.removeEventListener('keydown', handleKeydown);
      destroy();
    }
  };
}
