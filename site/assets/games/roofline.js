/* ============================================================
   MLSys Playground — Roofline Runner
   ------------------------------------------------------------
   Kernels fly in from the right at various heights. You control
   a horizontal catcher; move it up/down to match the kernel's
   height. Catching a kernel BELOW the ceiling: +score. Catching
   one ABOVE the ceiling: -score (you picked an unrealisable
   implementation). Missing: -life.

   30-second round. Score = kernels caught.
   ============================================================ */

window.MLSP = window.MLSP || {};
MLSP.games = MLSP.games || {};

MLSP.games.roofline = function(canvas, opts) {
  opts = opts || {};
  var ctx = canvas.getContext("2d");
  var W = canvas.width, H = canvas.height;

  var TIME_LIMIT_MS = 30000;
  var MAX_LIVES = 3;

  // Chart region
  var chartX = 80, chartY = 70, chartW = W - 140, chartH = H - 160;
  var chartRight = chartX + chartW;
  var chartBottom = chartY + chartH;

  // Roofline curve: memory-bound slope on left, compute-bound flat on right
  // Translate intensity x in [0, 1] to y (performance ceiling)
  var ridgeX = 0.45;          // position of ridge point
  var peakY = 0.9;            // peak performance (top of chart area, normalized)
  function ceilingAt(xFrac) {
    if (xFrac <= ridgeX) return (xFrac / ridgeX) * peakY;
    return peakY;
  }
  function yFromFrac(f) { return chartBottom - f * chartH; }
  function xFromFrac(f) { return chartX + f * chartW; }

  var ops = ["GEMM", "attn", "softmax", "conv", "elem", "relu", "layernorm", "gelu"];

  var state = {
    catcherX: chartX + chartW * 0.5,
    catcherY: chartBottom - 20,
    kernels: [],
    score: 0,
    lives: MAX_LIVES,
    timeLeft: TIME_LIMIT_MS,
    spawnCooldown: 700,
    over: false,
    won: false,
    shakeAmt: 0, shakeT: 0,
    particles: [], floats: []
  };

  var alltimeBest = MLSP.bestScore.get("roofline");
  var rand = Math.random;

  /* Input: arrow keys + mouse/touch */
  var keysDown = {};
  window.addEventListener("keydown", function(e) {
    if (!MLSP.inViewport(canvas)) return;
    keysDown[e.key] = true;
    if (e.key === "r" || e.key === "R") { e.preventDefault(); if (opts.onRetry) opts.onRetry(); }
    if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", " "].indexOf(e.key) >= 0) e.preventDefault();
  });
  window.addEventListener("keyup", function(e) { keysDown[e.key] = false; });
  canvas.addEventListener("mousemove", function(e) {
    if (state.over) return;
    var p = MLSP.canvasPoint(canvas, e);
    state.catcherX = Math.max(chartX, Math.min(chartRight, p.x));
    state.catcherY = Math.max(chartY, Math.min(chartBottom, p.y));
  });
  canvas.addEventListener("pointerdown", function(e) {
    e.preventDefault();
    if (state.over) { if (opts.onRetry) opts.onRetry(); return; }
    var p = MLSP.canvasPoint(canvas, e);
    state.catcherX = p.x;
    state.catcherY = p.y;
  });
  canvas.addEventListener("touchmove", function(e) {
    e.preventDefault();
    if (state.over) return;
    var p = MLSP.canvasPoint(canvas, e);
    state.catcherX = Math.max(chartX, Math.min(chartRight, p.x));
    state.catcherY = Math.max(chartY, Math.min(chartBottom, p.y));
  }, { passive: false });

  function spawnKernel() {
    var xFrac = 0.1 + rand() * 0.85;
    // Kernel true y: sometimes under ceiling, sometimes above
    var ceilFrac = ceilingAt(xFrac);
    var belowCeiling = rand() < 0.7;
    var yFrac = belowCeiling
      ? 0.1 + rand() * Math.max(0.05, ceilFrac - 0.1)
      : Math.min(0.95, ceilFrac + 0.05 + rand() * 0.25);
    state.kernels.push({
      x: chartRight + 30,
      y: yFromFrac(yFrac),
      targetX: xFromFrac(xFrac),
      xFrac: xFrac,
      yFrac: yFrac,
      belowCeiling: belowCeiling,
      op: ops[Math.floor(rand() * ops.length)],
      vx: -(1.0 + rand() * 0.7),
      state: "flying" // flying, caught, missed
    });
  }

  function updateKernels(dt) {
    var catcherR = 20;
    for (var i = 0; i < state.kernels.length; i++) {
      var k = state.kernels[i];
      if (k.state !== "flying") continue;
      k.x += k.vx * (dt * 0.15);
      // Arrived at its target x position
      if (k.x <= k.targetX + 2) {
        var dx = k.targetX - state.catcherX;
        var dy = k.y - state.catcherY;
        var dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < catcherR) {
          if (k.belowCeiling) {
            state.score++;
            addFloat(k.targetX, k.y - 10, "+1 " + k.op, "#3d9e5a");
            burst(k.targetX, k.y, "#3d9e5a", 8);
          } else {
            state.score = Math.max(0, state.score - 1);
            shake(6, 200);
            addFloat(k.targetX, k.y - 10, "above ceiling!", "#c44");
            burst(k.targetX, k.y, "#c44", 10);
          }
          k.state = "caught";
        } else {
          // Missed
          state.lives--;
          shake(4, 180);
          addFloat(k.targetX, k.y - 10, "missed " + k.op, "#c44");
          burst(k.targetX, k.y, "#c44", 5);
          k.state = "missed";
          if (state.lives <= 0 && !state.over) endGame(false);
        }
      }
    }
    state.kernels = state.kernels.filter(function(k){ return k.state === "flying"; });
  }

  function endGame(won) {
    state.over = true;
    state.won = won;
    if (state.score > alltimeBest) {
      alltimeBest = state.score;
      MLSP.bestScore.set("roofline", alltimeBest);
    }
    if (opts.onGameOver) opts.onGameOver({
      score: state.score,
      won: won,
      alltimeBest: alltimeBest,
      timeUsed: Math.round((TIME_LIMIT_MS - state.timeLeft) / 1000)
    });
  }

  function shake(a, ms) { state.shakeAmt = Math.max(state.shakeAmt, a); state.shakeT = Math.max(state.shakeT, ms); }
  function burst(x, y, color, n) {
    for (var i = 0; i < n; i++) {
      var ang = rand() * Math.PI * 2, spd = 1 + rand() * 2;
      state.particles.push({ x: x, y: y, vx: Math.cos(ang)*spd, vy: Math.sin(ang)*spd, age: 0, maxAge: 600, color: color });
    }
  }
  function addFloat(x, y, t, c) { state.floats.push({ x: x, y: y, text: t, color: c, age: 0, maxAge: 1000 }); }

  var lastTime = 0;
  function frame(now) {
    var dt = lastTime ? (now - lastTime) : 16;
    lastTime = now;
    if (dt > 100) dt = 100;

    if (!state.over) {
      state.timeLeft -= dt;
      if (state.timeLeft <= 0) { state.timeLeft = 0; endGame(true); }
      state.spawnCooldown -= dt;
      if (state.spawnCooldown <= 0) {
        spawnKernel();
        var elapsed = TIME_LIMIT_MS - state.timeLeft;
        state.spawnCooldown = 750 - Math.min(400, elapsed / 40);
      }
      // Keyboard movement
      if (keysDown["ArrowLeft"])  state.catcherX = Math.max(chartX, state.catcherX - 5);
      if (keysDown["ArrowRight"]) state.catcherX = Math.min(chartRight, state.catcherX + 5);
      if (keysDown["ArrowUp"])    state.catcherY = Math.max(chartY, state.catcherY - 5);
      if (keysDown["ArrowDown"])  state.catcherY = Math.min(chartBottom, state.catcherY + 5);
      updateKernels(dt);
    }

    state.shakeT = Math.max(0, state.shakeT - dt);
    if (state.shakeT === 0) state.shakeAmt = 0;
    for (var pp of state.particles) { pp.x += pp.vx; pp.y += pp.vy; pp.vy += 0.15; pp.age += dt; }
    state.particles = state.particles.filter(function(x){ return x.age < x.maxAge; });
    for (var ff of state.floats) { ff.age += dt; ff.y -= dt * 0.03; }
    state.floats = state.floats.filter(function(x){ return x.age < x.maxAge; });

    if (opts.onScoreChange && !state.over) opts.onScoreChange({
      score: state.score, lives: state.lives, timeLeft: state.timeLeft, alltimeBest: alltimeBest
    });

    draw();
    requestAnimationFrame(frame);
  }

  function draw() {
    var sx = 0, sy = 0;
    if (state.shakeAmt > 0) { sx = (Math.random()-0.5)*state.shakeAmt; sy = (Math.random()-0.5)*state.shakeAmt; }
    ctx.save();
    ctx.translate(sx, sy);
    ctx.clearRect(-20, -20, W + 40, H + 40);

    // Header
    ctx.fillStyle = "#333";
    ctx.font = "bold 15px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Roofline Runner", W/2, 24);
    ctx.font = "10.5px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#888";
    ctx.fillText("catch kernels UNDER the ceiling · avoid the ones above", W/2, 42);

    // Chart grid
    ctx.strokeStyle = "#eee";
    ctx.lineWidth = 1;
    for (var g = 0; g <= 4; g++) {
      var y = chartY + (g / 4) * chartH;
      ctx.beginPath(); ctx.moveTo(chartX, y); ctx.lineTo(chartRight, y); ctx.stroke();
      var x = chartX + (g / 4) * chartW;
      ctx.beginPath(); ctx.moveTo(x, chartY); ctx.lineTo(x, chartBottom); ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = "#555";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(chartX, chartY); ctx.lineTo(chartX, chartBottom); ctx.lineTo(chartRight, chartBottom);
    ctx.stroke();
    ctx.fillStyle = "#555";
    ctx.font = "10px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("arithmetic intensity (FLOPs / byte) →", (chartX + chartRight) / 2, chartBottom + 26);
    ctx.save();
    ctx.translate(chartX - 28, (chartY + chartBottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("performance (TFLOP/s) →", 0, 0);
    ctx.restore();

    // Region labels
    ctx.fillStyle = "#bbb";
    ctx.font = "italic 10px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("memory-bound", xFromFrac(ridgeX/2), chartY + 16);
    ctx.fillText("compute-bound", xFromFrac((ridgeX + 1) / 2), chartY + 16);

    // Roofline ceiling
    ctx.strokeStyle = "#a31f34";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xFromFrac(0), yFromFrac(0));
    ctx.lineTo(xFromFrac(ridgeX), yFromFrac(peakY));
    ctx.lineTo(xFromFrac(1), yFromFrac(peakY));
    ctx.stroke();

    // Ridge marker
    ctx.fillStyle = "#a31f34";
    ctx.beginPath();
    ctx.arc(xFromFrac(ridgeX), yFromFrac(peakY), 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText("ridge", xFromFrac(ridgeX), yFromFrac(peakY) - 8);

    // Above-ceiling fill
    ctx.fillStyle = "rgba(196,68,68,0.05)";
    ctx.beginPath();
    ctx.moveTo(xFromFrac(0), chartY);
    ctx.lineTo(xFromFrac(ridgeX), yFromFrac(peakY));
    ctx.lineTo(xFromFrac(1), yFromFrac(peakY));
    ctx.lineTo(xFromFrac(1), chartY);
    ctx.closePath();
    ctx.fill();

    // Kernels
    for (var i = 0; i < state.kernels.length; i++) {
      var k = state.kernels[i];
      ctx.fillStyle = k.belowCeiling ? "#4a90c4" : "#c44";
      ctx.strokeStyle = k.belowCeiling ? "#3d79a8" : "#a31f34";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(k.x, k.y, 7, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#fff";
      ctx.font = "bold 8px 'Helvetica Neue', Arial, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(k.op, k.x, k.y + 2.5);
    }

    // Catcher — a crosshair
    ctx.strokeStyle = "#a31f34";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(state.catcherX, state.catcherY, 16, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(state.catcherX - 22, state.catcherY); ctx.lineTo(state.catcherX - 10, state.catcherY);
    ctx.moveTo(state.catcherX + 10, state.catcherY); ctx.lineTo(state.catcherX + 22, state.catcherY);
    ctx.moveTo(state.catcherX, state.catcherY - 22); ctx.lineTo(state.catcherX, state.catcherY - 10);
    ctx.moveTo(state.catcherX, state.catcherY + 10); ctx.lineTo(state.catcherX, state.catcherY + 22);
    ctx.stroke();

    // Particles + floats
    for (var pi = 0; pi < state.particles.length; pi++) {
      var pa = state.particles[pi];
      ctx.globalAlpha = Math.max(0, 1 - pa.age / pa.maxAge);
      ctx.fillStyle = pa.color;
      ctx.beginPath(); ctx.arc(pa.x, pa.y, 2, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
    for (var fi = 0; fi < state.floats.length; fi++) {
      var ff = state.floats[fi];
      ctx.globalAlpha = Math.max(0, 1 - ff.age / ff.maxAge);
      ctx.fillStyle = ff.color;
      ctx.font = "bold 11px 'Helvetica Neue', Arial, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(ff.text, ff.x, ff.y);
    }
    ctx.globalAlpha = 1;

    drawHud();
    if (state.over) drawGameOver();
    ctx.restore();
  }

  function drawHud() {
    ctx.font = "11px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#333";
    ctx.textAlign = "left";
    ctx.fillText("score " + state.score, 20, H - 26);
    // Lives
    ctx.textAlign = "center";
    var lives = "";
    for (var i = 0; i < MAX_LIVES; i++) lives += i < state.lives ? "❤️ " : "🖤 ";
    ctx.fillText(lives, W/2 - 30, H - 26);
    // Time
    var secs = Math.ceil(state.timeLeft / 1000);
    ctx.fillStyle = secs <= 5 ? "#c44" : "#333";
    ctx.font = "bold 13px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("⏱ " + secs + "s", W/2 + 30, H - 26);
    ctx.fillStyle = "#333";
    ctx.font = "11px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("alltime best " + alltimeBest, W - 20, H - 26);
    ctx.font = "9px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#999";
    ctx.textAlign = "left";
    ctx.fillText("mouse or arrow keys to move the crosshair · R to retry", 20, H - 10);
  }

  function drawGameOver() {
    ctx.fillStyle = "rgba(255,255,255,0.92)";
    ctx.fillRect(0, 0, W, H);
    var color = state.won ? "#3d9e5a" : "#a31f34";
    var title = state.won ? "🏆 time!" : "out of lives";
    ctx.fillStyle = color;
    ctx.font = "bold 24px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(title, W/2, H/2 - 20);
    ctx.fillStyle = "#333";
    ctx.font = "14px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("caught " + state.score + " kernels · best " + alltimeBest, W/2, H/2 + 8);
    ctx.fillStyle = "#777";
    ctx.font = "italic 11px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("tap or press R to retry", W/2, H/2 + 32);
  }

  requestAnimationFrame(frame);

  return {
    id: "roofline",
    name: "Roofline Runner",
    ahaLabel: "You just played at",
    ahaText: "The roofline model. Every kernel has an arithmetic intensity (FLOPs per byte of memory traffic); the red line is the hardware's performance ceiling, which bends at the 'ridge' where workloads transition from memory-bound to compute-bound. You were catching kernels beneath the line because no amount of engineering can push above it.",
    buildShareText: function(r) {
      return "MLSys Playground · Roofline Runner\n" +
             "caught " + r.score + " kernels" + (r.won ? " 🏆" : "") + "\n" +
             "play → mlsysbook.ai/games/roofline/";
    }
  };
};
