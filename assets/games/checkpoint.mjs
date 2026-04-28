import * as runtime from "./runtime.mjs";

window.MLSP = window.MLSP || {};
window.MLSP.games = window.MLSP.games || {};

window.MLSP.games.checkpoint = async function(canvas, callbacks) {
  const { app, stage, width, height, PIXI, onTick, destroy } = await runtime.mountPixiOnCanvas(canvas, { bg: 0x111111 });
  
  const state = {
    score: 0,
    gameOver: false,
    started: false
  };
  
  const container = new PIXI.Container();
  stage.addChild(container);
  
  // Progress bar outline
  const barOutline = new PIXI.Graphics();
  barOutline.rect(50, height/2 - 20, width - 100, 40);
  barOutline.stroke({ color: 0xffffff, width: 2 });
  container.addChild(barOutline);
  
  const barFill = new PIXI.Graphics();
  container.addChild(barFill);
  const instruction = new PIXI.Text({
    text: "Hold Space to train. Release to write a checkpoint before a node failure.",
    style: { fill: 0xffffff, fontSize: 16, fontFamily: "Helvetica Neue, Arial", align: "center" }
  });
  instruction.anchor.set(0.5);
  instruction.position.set(width / 2, height / 2 - 62);
  container.addChild(instruction);
  const checkpointText = new PIXI.Text({
    text: "last checkpoint: 0%",
    style: { fill: 0xaaaaaa, fontSize: 13, fontFamily: "Helvetica Neue, Arial" }
  });
  checkpointText.anchor.set(0.5);
  checkpointText.position.set(width / 2, height / 2 + 46);
  container.addChild(checkpointText);
  
  let progress = 0;
  let lastCheckpoint = 0;
  let spaceHeld = false;
  
  let strikeTimer = 2000 + Math.random() * 3000;
  let warningPhase = false;
  
  const checkpoints = [];
  
  const downHandler = (e) => { if (state.started && e.code === 'Space') { e.preventDefault(); spaceHeld = true; } };
  const upHandler = (e) => { 
    if (e.code === 'Space') {
      e.preventDefault();
      spaceHeld = false;
      // Write checkpoint
      if (progress > lastCheckpoint && progress < width - 100 && !state.gameOver) {
        lastCheckpoint = progress;
        checkpointText.text = "last checkpoint: " + Math.floor((lastCheckpoint / (width - 100)) * 100) + "%";
        const cp = new PIXI.Graphics();
        cp.rect(50 + progress - 2, height/2 - 30, 4, 60);
        cp.fill({ color: 0x00ff00 });
        container.addChild(cp);
        checkpoints.push(cp);
        runtime.pop(stage, 50 + progress, height/2, 0x00ff00);
        runtime.floatText(stage, 50 + progress, height/2 - 40, "CHECKPOINT", 0x00ff00);
      }
    }
  };
  
  window.addEventListener('keydown', downHandler);
  window.addEventListener('keyup', upHandler);
  
  const bgWarning = new PIXI.Graphics();
  bgWarning.rect(0,0,width,height);
  bgWarning.fill({color: 0xff0000});
  bgWarning.alpha = 0;
  stage.addChildAt(bgWarning, 0);

  // Pre-game READY overlay
  runtime.mountReadyOverlay(stage, {
    width: width, height: height,
    title: "CHECKPOINT ROULETTE",
    goal: "Train fast. Checkpoint before a node failure strikes.",
    controls: "HOLD SPACE  train · RELEASE  write a checkpoint",
    onLaunch: () => { state.started = true; }
  });

  onTick((dt) => {
    if (!state.started) return;
    if (state.gameOver) return;

    if (spaceHeld) {
      progress += dt * 0.04; // 0.04 px per ms
      if (progress >= width - 100) {
        progress = width - 100;
        state.score = 100;
        callbacks.onScoreChange(state);
        state.gameOver = true;
        endGame(true);
      }
    }
    
    // Draw fill
    barFill.clear();
    if (progress > 0) {
      barFill.rect(50, height/2 - 20, progress, 40);
      barFill.fill({ color: spaceHeld ? 0x4a90c4 : 0x555555 });
    }
    
    state.score = Math.floor((progress / (width - 100)) * 100);
    callbacks.onScoreChange(state);
    
    strikeTimer -= dt;
    if (strikeTimer < 800 && !warningPhase) {
      warningPhase = true;
    }
    
    if (warningPhase) {
      bgWarning.alpha = 0.15 + Math.sin(performance.now() / 50) * 0.15;
    } else {
      bgWarning.alpha = 0;
    }
    
    if (strikeTimer <= 0) {
      // Strike!
      runtime.flash(stage, 0xffffff, 400, 0.9);
      runtime.shake(container, 20, 400);
      
      if (spaceHeld) {
        // Node failure while training! Reset to checkpoint
        progress = lastCheckpoint;
        runtime.floatText(stage, 50 + progress, height/2, "NODE FAILED!", 0xff0000, {size: 32});
      } else {
        // Safe
        runtime.floatText(stage, width/2, height/2 - 80, "DODGED", 0xaaaaaa);
      }
      
      strikeTimer = 1500 + Math.random() * 4000;
      warningPhase = false;
    }
  });
  
  function endGame(win) {
    const go = new PIXI.Text({ text: win ? "TRAINING COMPLETE" : "GAME OVER", style: { fill: win ? 0x00ff00 : 0xff0000, fontSize: 48, align: 'center' } });
    go.anchor.set(0.5);
    go.position.set(width/2, height/2 - 80);
    stage.addChild(go);
    callbacks.onGameOver({ score: state.score });
  }
  
  return {
    ahaLabel: "Fault Tolerance",
    ahaText: "Frequent checkpoints save progress during node failures, but taking them pauses training.",
    ahaLink: { href: "/", label: "Read Vol II: Fault Tolerance" },
    destroy: () => {
      window.removeEventListener('keydown', downHandler);
      window.removeEventListener('keyup', upHandler);
      destroy();
    }
  };
};