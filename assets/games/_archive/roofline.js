/* ============================================================
   MLSys Playground — Roofline Runner (v3)
   Catch kernels under the ceiling. Op intensity bands give each
   op type its proper position on the chart. Predictive landing
   reticle. Now with shared juice + emoji-grid share + cite-correct
   aha card.
   ============================================================ */

window.MLSP = window.MLSP || {};
MLSP.games = MLSP.games || {};

MLSP.games.roofline = function(canvas, opts) {
  opts = opts || {};
  var ctx = canvas.getContext("2d");
  var W = canvas.width, H = canvas.height;

  var TIME_LIMIT_MS = 30000;
  var MAX_LIVES = 3;

  function hashString(s) { var h = 2166136261 >>> 0; for (var i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)>>>0;} return h; }
  function mulberry32(seed) { var a = seed >>> 0; return function(){ a = (a + 0x6D2B79F5) >>> 0; var t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  var today = new Date().toISOString().slice(0, 10);
  var rand = mulberry32(hashString("roofline-" + today));

  var chartX = 80, chartY = 70, chartW = W - 140, chartH = H - 160;
  var chartRight = chartX + chartW, chartBottom = chartY + chartH;
  var ridgeX = 0.45, peakY = 0.9;
  function ceilingAt(xFrac) { return xFrac <= ridgeX ? (xFrac / ridgeX) * peakY : peakY; }
  function yFromFrac(f) { return chartBottom - f * chartH; }
  function xFromFrac(f) { return chartX + f * chartW; }

  var OPS = [
    { name: "GEMM",      band: [0.65, 0.85] },
    { name: "conv",      band: [0.50, 0.65] },
    { name: "attn",      band: [0.35, 0.50] },
    { name: "gelu",      band: [0.18, 0.28] },
    { name: "layernorm", band: [0.12, 0.20] },
    { name: "softmax",   band: [0.10, 0.18] },
    { name: "elem",      band: [0.05, 0.12] }
  ];

  var state = {
    catcherX: chartX + chartW * 0.5,
    catcherY: chartBottom - 20,
    kernels: [],
    score: 0, lives: MAX_LIVES,
    timeLeft: TIME_LIMIT_MS,
    spawnCooldown: 700,
    over: false, won: false,
    shakeAmt: 0, shakeT: 0,
    particles: [], floats: [], pops: [], flash: null,
    history: []   // for emoji grid share
  };
  var alltimeBest = MLSP.bestScore.get("roofline");

  var keysDown = {};
  window.addEventListener("keydown", function(e) {
    if (!MLSP.inViewport(canvas)) return;
    keysDown[e.key] = true;
    if (e.key === "r" || e.key === "R") { e.preventDefault(); if (opts.onRetry) opts.onRetry(); }
    if (["ArrowLeft","ArrowRight","ArrowUp","ArrowDown"," "].indexOf(e.key) >= 0) e.preventDefault();
  });
  window.addEventListener("keyup", function(e) { keysDown[e.key] = false; });
  canvas.addEventListener("mousemove", function(e) { if (state.over) return; var p = MLSP.canvasPoint(canvas, e); state.catcherX = Math.max(chartX, Math.min(chartRight, p.x)); state.catcherY = Math.max(chartY, Math.min(chartBottom, p.y)); });
  canvas.addEventListener("pointerdown", function(e) { e.preventDefault(); if (state.over) { if (opts.onRetry) opts.onRetry(); return; } var p = MLSP.canvasPoint(canvas, e); state.catcherX = p.x; state.catcherY = p.y; });
  canvas.addEventListener("touchmove", function(e) {
    e.preventDefault(); if (state.over) return;
    var p = MLSP.canvasPoint(canvas, e);
    // Y-offset so catcher is visible above touch finger (per accessibility review)
    state.catcherX = Math.max(chartX, Math.min(chartRight, p.x));
    state.catcherY = Math.max(chartY, Math.min(chartBottom, p.y - 40));
  }, { passive: false });

  function spawnKernel() {
    var op = OPS[Math.floor(rand() * OPS.length)];
    var xFrac = op.band[0] + rand() * (op.band[1] - op.band[0]);
    var ceilFrac = ceilingAt(xFrac);
    var belowCeiling = rand() < 0.7;
    var yFrac = belowCeiling
      ? 0.08 + rand() * Math.max(0.05, ceilFrac - 0.1)
      : Math.min(0.95, ceilFrac + 0.05 + rand() * 0.25);
    state.kernels.push({
      x: chartRight + 30, y: yFromFrac(yFrac),
      targetX: xFromFrac(xFrac), xFrac: xFrac, yFrac: yFrac,
      belowCeiling: belowCeiling, op: op.name,
      vx: -(0.85 + rand() * 0.55), state: "flying"
    });
  }

  function updateKernels(dt) {
    var catcherR = 22;
    for (var i = 0; i < state.kernels.length; i++) {
      var k = state.kernels[i];
      if (k.state !== "flying") continue;
      var prevX = k.x;
      k.x += k.vx * (dt * 0.15);
      // Resolve atomically the frame the kernel crosses its targetX — no
      // mercy window. The ghost reticle promises 'land here'; the catcher
      // must actually be there when the kernel arrives.
      if (prevX > k.targetX && k.x <= k.targetX) {
        var dx = k.targetX - state.catcherX;
        var dy = k.y - state.catcherY;
        var dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < catcherR) {
          if (k.belowCeiling) {
            state.score++;
            state.history.push("good");
            addFloat(k.targetX, k.y - 10, "+1 " + k.op, "#3d9e5a");
            burst(k.targetX, k.y, "#3d9e5a", 8);
            MLSP.pop(state, k.targetX, k.y, "#3d9e5a", { r: 18 });
          } else {
            state.score = Math.max(0, state.score - 1);
            state.history.push("bad");
            shake(6, 200);
            addFloat(k.targetX, k.y - 10, "above ceiling!", "#c44");
            burst(k.targetX, k.y, "#c44", 10);
            MLSP.pop(state, k.targetX, k.y, "#c44", { r: 22 });
          }
          k.state = "caught";
        } else {
          state.lives--;
          state.history.push("miss");
          shake(4, 180);
          addFloat(k.targetX, k.y - 10, "missed " + k.op, "#c44");
          burst(k.targetX, k.y, "#c44", 5);
          k.state = "missed";
          if (state.lives <= 0 && !state.over) {
            MLSP.flash(state, "#a31f34", 320);
            endGame(false);
          }
        }
      }
    }
    state.kernels = state.kernels.filter(function(k){ return k.state === "flying"; });
  }

  function endGame(won) {
    state.over = true; state.won = won;
    if (won) MLSP.flash(state, "#3d9e5a", 360);
    if (state.score > alltimeBest) { alltimeBest = state.score; MLSP.bestScore.set("roofline", alltimeBest); }
    if (opts.onGameOver) opts.onGameOver({
      score: state.score, won: won, alltimeBest: alltimeBest,
      emojiGrid: buildEmojiGrid()
    });
  }

  function buildEmojiGrid() {
    // Last 12 outcomes as a row: 🟦 caught-good, 🟥 caught-bad, ⬛ missed
    var slice = state.history.slice(-12);
    return slice.map(function(s){ return s === "good" ? "🟦" : s === "bad" ? "🟥" : "⬛"; }).join("");
  }

  function shake(a, ms) { state.shakeAmt = Math.max(state.shakeAmt, a); state.shakeT = Math.max(state.shakeT, ms); }
  function burst(x, y, color, n) { for (var i = 0; i < n; i++) { var ang = rand() * Math.PI * 2, spd = 1 + rand() * 2; state.particles.push({ x: x, y: y, vx: Math.cos(ang)*spd, vy: Math.sin(ang)*spd, age: 0, maxAge: 600, color: color }); } }
  function addFloat(x, y, t, c) { state.floats.push({ x: x, y: y, text: t, color: c, age: 0, maxAge: 1000 }); }

  var lastTime = 0;
  function frame(now) {
    var dt = lastTime ? (now - lastTime) : 16; lastTime = now; if (dt > 100) dt = 100;
    if (!state.over) {
      state.timeLeft -= dt;
      if (state.timeLeft <= 0) { state.timeLeft = 0; endGame(true); }
      state.spawnCooldown -= dt;
      if (state.spawnCooldown <= 0) { spawnKernel(); var elapsed = TIME_LIMIT_MS - state.timeLeft; state.spawnCooldown = 800 - Math.min(450, elapsed / 38); }
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
    MLSP.tickJuice(state, dt);

    if (opts.onScoreChange && !state.over) opts.onScoreChange({ score: state.score, lives: state.lives, timeLeft: state.timeLeft, alltimeBest: alltimeBest });
    draw();
    requestAnimationFrame(frame);
  }

  function draw() {
    var sx = 0, sy = 0;
    if (state.shakeAmt > 0) { sx = (rand()-0.5)*state.shakeAmt; sy = (rand()-0.5)*state.shakeAmt; }
    ctx.save(); ctx.translate(sx, sy);
    ctx.clearRect(-20, -20, W + 40, H + 40);

    ctx.fillStyle = "#333"; ctx.font = "bold 15px 'Helvetica Neue', Arial, sans-serif"; ctx.textAlign = "center";
    ctx.fillText("Roofline Runner", W/2, 24);
    ctx.font = "10.5px 'Helvetica Neue', Arial, sans-serif"; ctx.fillStyle = "#888";
    ctx.fillText("catch kernels UNDER the red ceiling · the dotted ghost shows where each will land", W/2, 42);

    ctx.strokeStyle = "#eee"; ctx.lineWidth = 1;
    for (var g = 0; g <= 4; g++) {
      var y = chartY + (g / 4) * chartH;
      ctx.beginPath(); ctx.moveTo(chartX, y); ctx.lineTo(chartRight, y); ctx.stroke();
      var x = chartX + (g / 4) * chartW;
      ctx.beginPath(); ctx.moveTo(x, chartY); ctx.lineTo(x, chartBottom); ctx.stroke();
    }
    ctx.strokeStyle = "#555"; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(chartX, chartY); ctx.lineTo(chartX, chartBottom); ctx.lineTo(chartRight, chartBottom); ctx.stroke();
    ctx.fillStyle = "#555"; ctx.font = "10px 'Helvetica Neue', Arial, sans-serif"; ctx.textAlign = "center";
    ctx.fillText("arithmetic intensity (FLOPs / byte) →", (chartX + chartRight) / 2, chartBottom + 26);
    ctx.save(); ctx.translate(chartX - 28, (chartY + chartBottom) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("performance (TFLOP/s) →", 0, 0); ctx.restore();
    ctx.fillStyle = "#bbb"; ctx.font = "italic 10px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("memory-bound", xFromFrac(ridgeX/2), chartY + 16);
    ctx.fillText("compute-bound", xFromFrac((ridgeX + 1) / 2), chartY + 16);
    ctx.strokeStyle = "#a31f34"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(xFromFrac(0), yFromFrac(0)); ctx.lineTo(xFromFrac(ridgeX), yFromFrac(peakY)); ctx.lineTo(xFromFrac(1), yFromFrac(peakY)); ctx.stroke();
    ctx.fillStyle = "#a31f34"; ctx.beginPath(); ctx.arc(xFromFrac(ridgeX), yFromFrac(peakY), 4, 0, Math.PI * 2); ctx.fill();
    ctx.fillText("ridge", xFromFrac(ridgeX), yFromFrac(peakY) - 8);
    ctx.fillStyle = "rgba(196,68,68,0.05)";
    ctx.beginPath(); ctx.moveTo(xFromFrac(0), chartY); ctx.lineTo(xFromFrac(ridgeX), yFromFrac(peakY)); ctx.lineTo(xFromFrac(1), yFromFrac(peakY)); ctx.lineTo(xFromFrac(1), chartY); ctx.closePath(); ctx.fill();

    for (var i = 0; i < state.kernels.length; i++) {
      var k = state.kernels[i];
      if (k.state !== "flying") continue;
      ctx.strokeStyle = k.belowCeiling ? "rgba(74,144,196,0.4)" : "rgba(196,68,68,0.4)";
      ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.arc(k.targetX, k.y, 10, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
    }
    for (var i = 0; i < state.kernels.length; i++) {
      var k = state.kernels[i];
      ctx.fillStyle = k.belowCeiling ? "#4a90c4" : "#c44";
      ctx.strokeStyle = k.belowCeiling ? "#3d79a8" : "#a31f34";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(k.x, k.y, 7, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#fff"; ctx.font = "bold 8px 'Helvetica Neue', Arial, sans-serif"; ctx.textAlign = "center";
      ctx.fillText(k.op, k.x, k.y + 2.5);
    }

    ctx.strokeStyle = "#a31f34"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(state.catcherX, state.catcherY, 16, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(state.catcherX - 22, state.catcherY); ctx.lineTo(state.catcherX - 10, state.catcherY);
    ctx.moveTo(state.catcherX + 10, state.catcherY); ctx.lineTo(state.catcherX + 22, state.catcherY);
    ctx.moveTo(state.catcherX, state.catcherY - 22); ctx.lineTo(state.catcherX, state.catcherY - 10);
    ctx.moveTo(state.catcherX, state.catcherY + 10); ctx.lineTo(state.catcherX, state.catcherY + 22);
    ctx.stroke();

    for (var pi = 0; pi < state.particles.length; pi++) { var pa = state.particles[pi]; ctx.globalAlpha = Math.max(0, 1 - pa.age / pa.maxAge); ctx.fillStyle = pa.color; ctx.beginPath(); ctx.arc(pa.x, pa.y, 2, 0, Math.PI * 2); ctx.fill(); }
    ctx.globalAlpha = 1;
    for (var fi = 0; fi < state.floats.length; fi++) { var ff = state.floats[fi]; ctx.globalAlpha = Math.max(0, 1 - ff.age / ff.maxAge); ctx.fillStyle = ff.color; ctx.font = "bold 11px 'Helvetica Neue', Arial, sans-serif"; ctx.textAlign = "center"; ctx.fillText(ff.text, ff.x, ff.y); }
    ctx.globalAlpha = 1;

    drawHud();
    MLSP.drawJuice(ctx, state, W, H);
    if (state.over) drawGameOver();
    ctx.restore();
  }

  function drawHud() {
    ctx.font = "11px 'Helvetica Neue', Arial, sans-serif"; ctx.fillStyle = "#333";
    ctx.textAlign = "left"; ctx.fillText("score " + state.score, 20, H - 26);
    ctx.textAlign = "center";
    var lives = ""; for (var i = 0; i < MAX_LIVES; i++) lives += i < state.lives ? "❤️ " : "🖤 ";
    ctx.fillText(lives, W/2 - 30, H - 26);
    var secs = Math.ceil(state.timeLeft / 1000);
    ctx.fillStyle = secs <= 5 ? "#c44" : "#333"; ctx.font = "bold 13px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("⏱ " + secs + "s", W/2 + 30, H - 26);
    ctx.fillStyle = "#333"; ctx.font = "11px 'Helvetica Neue', Arial, sans-serif"; ctx.textAlign = "right";
    ctx.fillText("all-time best " + alltimeBest, W - 20, H - 26);
    ctx.font = "9px 'Helvetica Neue', Arial, sans-serif"; ctx.fillStyle = "#999"; ctx.textAlign = "left";
    ctx.fillText("daily " + today + " · day " + MLSP.dayNumber() + " · mouse / arrows · R retry", 20, H - 10);
  }

  function drawGameOver() {
    ctx.fillStyle = "rgba(255,255,255,0.92)"; ctx.fillRect(0, 0, W, H);
    var color = state.won ? "#3d9e5a" : "#a31f34";
    var title = state.won ? "🏆 time!" : "out of lives";
    ctx.fillStyle = color; ctx.font = "bold 24px 'Helvetica Neue', Arial, sans-serif"; ctx.textAlign = "center";
    ctx.fillText(title, W/2, H/2 - 20);
    ctx.fillStyle = "#333"; ctx.font = "14px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("caught " + state.score + " kernels · best " + alltimeBest, W/2, H/2 + 8);
    ctx.fillStyle = "#777"; ctx.font = "italic 11px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("tap or press R to retry", W/2, H/2 + 32);
  }

  requestAnimationFrame(frame);

  return {
    id: "roofline",
    name: "Roofline Runner",
    ahaLabel: "You just played at",
    ahaText: "The roofline model (Williams, Waterman, Patterson 2009). Each operator has an arithmetic intensity (FLOPs per byte of memory traffic) — GEMM and conv sit on the right (compute-bound), elementwise and softmax on the left (memory-bound), attention spans both depending on sequence length and prefill-vs-decode phase. The ceiling is your hardware's hard limit. Real engineering raises a kernel's intensity (fusion, tiling, recompute) to push it under a higher ceiling — not catching what falls.",
    buildShareText: function(r) {
      return "MLSysBook Playground · Roofline Runner · day " + MLSP.dayNumber() + "\n" +
             "caught " + r.score + " kernels" + (r.won ? " 🏆" : " ✗") + "\n" +
             r.emojiGrid + "\n" +
             "play → mlsysbook.ai/games/roofline/";
    }
  };
};
