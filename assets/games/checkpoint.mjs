import * as runtime from "./runtime.mjs";

window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};

window.MLSP.games.checkpoint = async function(canvas, callbacks = {}) {
  const { stage, width: W, height: H, PIXI, onTick, destroy } =
    await runtime.mountPixiOnCanvas(canvas, { bg: 0x111827 });

  const RUN_MS = 60000;
  const TRAIN_MS = 20000;
  const WRITE_MS = 1100;
  const BAR_X = 50;
  const BAR_Y = H / 2 - 20;
  const BAR_W = W - 100;
  const state = {
    started: false,
    over: false,
    progress: 0,
    saved: 0,
    pending: 0,
    writeLeft: 0,
    timeLeft: RUN_MS,
    failureLeft: 2400 + Math.random() * 1800,
    failures: 0,
    checkpoints: 0,
    holding: false
  };

  const warning = new PIXI.Graphics();
  const bar = new PIXI.Graphics();
  const savedMark = new PIXI.Graphics();
  const writeBar = new PIXI.Graphics();
  const status = new PIXI.Text({
    text: "Hold Space or the button to train",
    style: { fill: 0xffffff, fontSize: 18, fontWeight: "600", align: "center" }
  });
  status.anchor.set(0.5);
  status.position.set(W / 2, BAR_Y - 58);

  const savedText = new PIXI.Text({
    text: "Saved progress  0%",
    style: { fill: 0xa7c4da, fontSize: 14 }
  });
  savedText.position.set(BAR_X, BAR_Y + 62);
  const timerText = new PIXI.Text({
    text: "Time  60s",
    style: { fill: 0xffffff, fontSize: 14 }
  });
  timerText.anchor.set(1, 0);
  timerText.position.set(W - BAR_X, BAR_Y + 62);
  const legend = new PIXI.Text({
    text: "Blue: live training   |   Green: durable checkpoint",
    style: { fill: 0xb6bec9, fontSize: 13, align: "center" }
  });
  legend.anchor.set(0.5);
  legend.position.set(W / 2, BAR_Y + 110);
  stage.addChild(warning, bar, savedMark, writeBar, status, savedText, timerText, legend);

  function reportProgress() {
    callbacks.onScoreChange?.({ score: Math.floor(state.progress), saved: Math.floor(state.saved) });
  }

  function draw() {
    bar.clear();
    bar.roundRect(BAR_X, BAR_Y, BAR_W, 40, 5).fill(0x263444);
    if (state.progress > 0) {
      bar.roundRect(BAR_X, BAR_Y, BAR_W * state.progress / 100, 40, 5)
        .fill(state.writeLeft > 0 ? 0xc87b2a : 0x4a90c4);
    }
    bar.roundRect(BAR_X, BAR_Y, BAR_W, 40, 5).stroke({ color: 0xdbe5ee, width: 2 });
    savedMark.clear();
    if (state.saved > 0) {
      const x = BAR_X + BAR_W * state.saved / 100;
      savedMark.moveTo(x, BAR_Y - 11).lineTo(x, BAR_Y + 51)
        .stroke({ color: 0x64d98b, width: 4 });
    }
    writeBar.clear();
    if (state.writeLeft > 0) {
      writeBar.roundRect(BAR_X, BAR_Y + 49, BAR_W * (1 - state.writeLeft / WRITE_MS), 4, 2)
        .fill(0xffc46b);
    }
    savedText.text = `Saved progress  ${Math.floor(state.saved)}%`;
    timerText.text = `Time  ${Math.ceil(state.timeLeft / 1000)}s`;
  }

  function startTraining() {
    if (!state.started || state.over || state.writeLeft > 0) return;
    state.holding = true;
    status.text = "Training — release to write a checkpoint";
  }

  function stopTraining() {
    if (!state.holding) return;
    state.holding = false;
    if (state.over || state.progress <= state.saved || state.writeLeft > 0) return;
    state.pending = state.progress;
    state.writeLeft = WRITE_MS;
    status.text = "Writing checkpoint — training paused";
  }

  const ready = runtime.mountReadyOverlay(stage, {
    width: W, height: H,
    title: "CHECKPOINT ROULETTE",
    goal: "Reach 100% before time runs out. A failure restores your last saved point.",
    controls: "HOLD SPACE or the button to train · RELEASE to save (1.1s)",
    onLaunch: () => {
      state.started = true;
      status.visible = true;
      reportProgress();
    }
  });
  // Hide the live instruction until the ready overlay is dismissed.
  status.visible = false;

  function finish(won) {
    if (state.over) return;
    state.over = true;
    state.holding = false;
    state.writeLeft = 0;
    const shade = new PIXI.Graphics();
    shade.rect(0, 0, W, H).fill({ color: 0x101827, alpha: 0.88 });
    const title = new PIXI.Text({
      text: won ? "TRAINING COMPLETE" : "TIME EXPIRED",
      style: { fill: won ? 0x64d98b : 0xff8b8b, fontSize: 32, fontWeight: "700" }
    });
    title.anchor.set(0.5);
    title.position.set(W / 2, H / 2 - 42);
    const summary = new PIXI.Text({
      text: `${state.checkpoints} checkpoints · ${state.failures} failures · ${Math.ceil((RUN_MS - state.timeLeft) / 1000)}s elapsed`,
      style: { fill: 0xffffff, fontSize: 16 }
    });
    summary.anchor.set(0.5);
    summary.position.set(W / 2, H / 2 + 8);
    const retry = new PIXI.Text({
      text: "Press R or tap to try again",
      style: { fill: 0xffd6a8, fontSize: 15 }
    });
    retry.anchor.set(0.5);
    retry.position.set(W / 2, H / 2 + 48);
    stage.addChild(shade, title, summary, retry);
    callbacks.onGameOver?.({
      won,
      score: Math.floor(state.progress),
      checkpoints: state.checkpoints,
      failures: state.failures,
      elapsedSeconds: Math.ceil((RUN_MS - state.timeLeft) / 1000)
    });
  }

  function fail() {
    state.failures++;
    state.progress = state.saved;
    state.pending = 0;
    state.writeLeft = 0;
    state.holding = false;
    status.text = "Node failed — restored last checkpoint";
    runtime.flash(stage, 0xc44444, 350, 0.45);
    reportProgress();
  }

  const downHandler = (e) => {
    if (e.code !== "Space") return;
    e.preventDefault();
    if (e.repeat) return;
    if (state.started) startTraining();
  };
  const upHandler = (e) => {
    if (e.code !== "Space") return;
    e.preventDefault();
    stopTraining();
  };
  const retryHandler = (e) => {
    if (state.over && e.key.toLowerCase() === "r") callbacks.onRetry?.();
  };
  const blurHandler = () => stopTraining();
  const canvasHandler = () => { if (state.over) callbacks.onRetry?.(); };
  window.addEventListener("keydown", downHandler);
  window.addEventListener("keyup", upHandler);
  window.addEventListener("keydown", retryHandler);
  window.addEventListener("blur", blurHandler);
  canvas.addEventListener("pointerdown", canvasHandler);

  const control = callbacks.controlButton;
  const controlDown = (e) => {
    e.preventDefault();
    if (!state.started) ready.dismiss();
    if (state.over) { callbacks.onRetry?.(); return; }
    control.setPointerCapture(e.pointerId);
    startTraining();
  };
  const controlUp = () => stopTraining();
  if (control) {
    control.addEventListener("pointerdown", controlDown);
    control.addEventListener("pointerup", controlUp);
    control.addEventListener("pointercancel", controlUp);
    control.addEventListener("lostpointercapture", controlUp);
  }

  onTick((dt) => {
    if (!state.started || state.over) return;
    const step = Math.min(dt, 100);
    state.timeLeft = Math.max(0, state.timeLeft - step);
    state.failureLeft -= step;

    if (state.writeLeft > 0) {
      state.writeLeft = Math.max(0, state.writeLeft - step);
      if (state.writeLeft === 0) {
        state.saved = state.pending;
        state.pending = 0;
        state.checkpoints++;
        status.text = "Checkpoint saved — hold to keep training";
        reportProgress();
      }
    } else if (state.holding) {
      state.progress = Math.min(100, state.progress + step * 100 / TRAIN_MS);
      reportProgress();
      if (state.progress >= 100) { finish(true); draw(); return; }
    }

    warning.clear();
    if (state.failureLeft < 1600) {
      warning.rect(0, 0, W, H).fill({ color: 0xc44444, alpha: 0.10 });
      if (state.writeLeft === 0 && !state.holding) status.text = "Failure imminent — hold after it passes";
    }
    if (state.failureLeft <= 0) {
      if (state.holding || state.writeLeft > 0) fail();
      else status.text = "Failure passed while idle — train or save when ready";
      state.failureLeft = 2200 + Math.random() * 2200;
      warning.clear();
    }
    if (state.timeLeft <= 0) finish(false);
    draw();
  });
  draw();

  return {
    ahaLabel: "Fault Tolerance",
    ahaText: "A completed checkpoint limits lost work after a node failure, but writing it pauses training and can itself be interrupted.",
    ahaLink: { href: "/vol2/fault_tolerance/fault_tolerance.html", label: "Read Volume II: Fault Tolerance" },
    destroy: () => {
      window.removeEventListener("keydown", downHandler);
      window.removeEventListener("keyup", upHandler);
      window.removeEventListener("keydown", retryHandler);
      window.removeEventListener("blur", blurHandler);
      canvas.removeEventListener("pointerdown", canvasHandler);
      if (control) {
        control.removeEventListener("pointerdown", controlDown);
        control.removeEventListener("pointerup", controlUp);
        control.removeEventListener("pointercancel", controlUp);
        control.removeEventListener("lostpointercapture", controlUp);
      }
      destroy();
    }
  };
};
