/* ============================================================
   MLSys Playground — OOM (v2, lifetime-driven)
   ------------------------------------------------------------
   Tetris-shaped, but the freeing mechanic is now ML-correct:
   blocks free on TRAINING STEP events, not Tetris-style row clears.

   Block types and lifetimes:
     - Activation (blue): freed when paired with its grad in the
       backward pass.
     - Gradient (red): all gradients clear on every "step!" event.
     - Optimizer state (orange): persistent across the run.
     - KV cache (green): persistent until "reset cache".

   The STEP! event fires on a tick (every ~7s) and visually
   clears every red gradient block simultaneously — your memory
   relief moment, but tied to real ML semantics instead of
   spatial accident.
   ============================================================ */

window.MLSP = window.MLSP || {};
MLSP.games = MLSP.games || {};

MLSP.games.oom = function(canvas, opts) {
  opts = opts || {};
  var ctx = canvas.getContext("2d");
  var W = canvas.width, H = canvas.height;

  var TIME_LIMIT_MS = 60000;
  var STEP_INTERVAL_MS = 7000;
  var BACKWARD_INTERVAL_MS = 4500;

  /* Daily seed */
  function hashString(s) { var h = 2166136261 >>> 0; for (var i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)>>>0;} return h; }
  function mulberry32(seed) { var a = seed >>> 0; return function(){ a = (a + 0x6D2B79F5) >>> 0; var t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  var today = new Date().toISOString().slice(0, 10);
  var rand = mulberry32(hashString("oom-" + today));

  // HBM region
  var hbmX = W/2 - 160, hbmY = 80, hbmW = 320, hbmH = H - 200;
  var cellSize = 20;
  var cols = Math.floor(hbmW / cellSize);
  var rows = Math.floor(hbmH / cellSize);

  var BLOCK_TYPES = [
    { name: "activation",      kind: "act",   color: "#cfe2f3", stroke: "#4a90c4", cells: [[0,0],[1,0],[2,0]],         freq: 0.35 },
    { name: "activation",      kind: "act",   color: "#cfe2f3", stroke: "#4a90c4", cells: [[0,0],[1,0]],               freq: 0.20 },
    { name: "gradient",        kind: "grad",  color: "#f9d6d5", stroke: "#c44",    cells: [[0,0],[0,1]],               freq: 0.20 },
    { name: "optimizer state", kind: "opt",   color: "#fdebd0", stroke: "#c87b2a", cells: [[0,0],[1,0],[0,1],[1,1]],   freq: 0.15 },
    { name: "KV cache",        kind: "kv",    color: "#d4edda", stroke: "#3d9e5a", cells: [[0,0],[0,1],[0,2]],         freq: 0.10 }
  ];
  // Build cumulative weights for sampling
  var freqSum = 0;
  for (var t of BLOCK_TYPES) freqSum += t.freq;

  // Each grid cell stores either null or an object: { kind, color, stroke, blockId, age }.
  // Tracking blockId lets us free entire blocks at once (not partial cells).
  var grid = [];
  for (var r = 0; r < rows; r++) {
    var row = [];
    for (var c = 0; c < cols; c++) row.push(null);
    grid.push(row);
  }
  var nextBlockId = 1;

  var state = {
    current: null,
    nextFallIn: 600,
    fallInterval: 800,
    score: 0,
    over: false,
    timeLeft: TIME_LIMIT_MS,
    stepCountdown: STEP_INTERVAL_MS,
    backwardCountdown: BACKWARD_INTERVAL_MS,
    stepFlashTime: 0,
    backwardFlashTime: 0,
    shakeAmt: 0, shakeT: 0,
    floats: [], particles: []
  };

  var alltimeBest = MLSP.bestScore.get("oom");

  function pickType() {
    var r = rand() * freqSum;
    for (var i = 0; i < BLOCK_TYPES.length; i++) {
      r -= BLOCK_TYPES[i].freq;
      if (r <= 0) return BLOCK_TYPES[i];
    }
    return BLOCK_TYPES[0];
  }

  function spawnBlock() {
    var type = pickType();
    var block = { type: type, row: 0, col: Math.floor(cols / 2) - 1, blockId: nextBlockId++ };
    if (collides(block, block.row, block.col)) {
      state.over = true;
      shake(10, 400);
      endGame();
      return;
    }
    state.current = block;
  }

  function collides(block, testRow, testCol) {
    for (var i = 0; i < block.type.cells.length; i++) {
      var dr = block.type.cells[i][1], dc = block.type.cells[i][0];
      var r = testRow + dr, c = testCol + dc;
      if (r < 0 || r >= rows || c < 0 || c >= cols) return true;
      if (grid[r][c] !== null) return true;
    }
    return false;
  }

  function lockBlock() {
    var b = state.current;
    if (!b) return;
    for (var i = 0; i < b.type.cells.length; i++) {
      var dr = b.type.cells[i][1], dc = b.type.cells[i][0];
      grid[b.row + dr][b.col + dc] = {
        kind: b.type.kind,
        color: b.type.color,
        stroke: b.type.stroke,
        blockId: b.blockId
      };
    }
    state.score++;
    addFloat(W/2, hbmY - 12, "+1 " + b.type.name, b.type.stroke);
    state.fallInterval = Math.max(280, 800 - state.score * 8);
    state.current = null;
    state.nextFallIn = 350;
  }

  function tryMove(dc, dr) {
    if (!state.current || state.over) return;
    var b = state.current;
    if (!collides(b, b.row + dr, b.col + dc)) { b.row += dr; b.col += dc; }
    else if (dr > 0) lockBlock();
  }

  function hardDrop() {
    if (!state.current || state.over) return;
    var b = state.current;
    while (!collides(b, b.row + 1, b.col)) b.row++;
    lockBlock();
  }

  /* ----- Lifetime-driven freeing (replaces row-clear) ----- */
  function fireStepEvent() {
    state.stepFlashTime = 1200;
    var freedBlockIds = {};
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var cell = grid[r][c];
        if (cell && cell.kind === "grad") {
          freedBlockIds[cell.blockId] = true;
          burst(hbmX + c*cellSize + cellSize/2, hbmY + r*cellSize + cellSize/2, "#c44", 3);
          grid[r][c] = null;
        }
      }
    }
    var n = Object.keys(freedBlockIds).length;
    if (n > 0) {
      state.score += n * 2;
      addFloat(W/2, H/2, "step()  · " + n + " gradients freed", "#3d9e5a");
    } else {
      addFloat(W/2, H/2, "step()", "#888");
    }
  }

  function fireBackwardEvent() {
    state.backwardFlashTime = 900;
    // Find the oldest activation blocks (lowest blockId of kind="act") and free 1-2 of them.
    var actIds = {};
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var cell = grid[r][c];
        if (cell && cell.kind === "act") actIds[cell.blockId] = true;
      }
    }
    var ids = Object.keys(actIds).map(Number).sort(function(a,b){ return a-b; });
    var toFree = ids.slice(0, 2);
    if (toFree.length === 0) return;
    var freedSet = {};
    toFree.forEach(function(id){ freedSet[id] = true; });
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var cell = grid[r][c];
        if (cell && freedSet[cell.blockId]) {
          burst(hbmX + c*cellSize + cellSize/2, hbmY + r*cellSize + cellSize/2, "#4a90c4", 2);
          grid[r][c] = null;
        }
      }
    }
    state.score += toFree.length;
    addFloat(W/2, H/2 + 24, "backward · " + toFree.length + " activations consumed", "#4a90c4");
  }

  function endGame() {
    if (state.score > alltimeBest) {
      alltimeBest = state.score;
      MLSP.bestScore.set("oom", alltimeBest);
    }
    if (opts.onGameOver) opts.onGameOver({ score: state.score, alltimeBest: alltimeBest });
  }

  function shake(a, ms) { state.shakeAmt = Math.max(state.shakeAmt, a); state.shakeT = Math.max(state.shakeT, ms); }
  function burst(x, y, color, n) {
    for (var i = 0; i < n; i++) {
      var ang = rand() * Math.PI * 2, spd = 1 + rand() * 2;
      state.particles.push({ x: x, y: y, vx: Math.cos(ang)*spd, vy: Math.sin(ang)*spd, age: 0, maxAge: 600, color: color });
    }
  }
  function addFloat(x, y, t, c) { state.floats.push({ x: x, y: y, text: t, color: c, age: 0, maxAge: 1200 }); }

  /* Input */
  window.addEventListener("keydown", function(e) {
    if (!MLSP.inViewport(canvas)) return;
    if (state.over) {
      if (e.key === "r" || e.key === "R") { e.preventDefault(); if (opts.onRetry) opts.onRetry(); }
      return;
    }
    if (e.key === "ArrowLeft")  { e.preventDefault(); tryMove(-1, 0); }
    if (e.key === "ArrowRight") { e.preventDefault(); tryMove(1, 0); }
    if (e.key === "ArrowDown")  { e.preventDefault(); tryMove(0, 1); }
    if (e.key === " ")          { e.preventDefault(); hardDrop(); }
    if (e.key === "r" || e.key === "R") { e.preventDefault(); if (opts.onRetry) opts.onRetry(); }
  });
  canvas.addEventListener("pointerdown", function(e) {
    e.preventDefault();
    if (state.over) { if (opts.onRetry) opts.onRetry(); return; }
    var p = MLSP.canvasPoint(canvas, e);
    if (!state.current) return;
    var b = state.current;
    var blockX = hbmX + b.col * cellSize;
    if (p.y > hbmY + b.row * cellSize + 20) { hardDrop(); return; }
    if (p.x < blockX + cellSize) tryMove(-1, 0);
    else if (p.x > blockX + cellSize) tryMove(1, 0);
  });

  var lastTime = 0;
  function frame(now) {
    var dt = lastTime ? (now - lastTime) : 16;
    lastTime = now;
    if (dt > 100) dt = 100;

    if (!state.over) {
      state.timeLeft -= dt;
      if (state.timeLeft <= 0) { state.timeLeft = 0; state.over = true; endGame(); }
      // STEP event
      state.stepCountdown -= dt;
      if (state.stepCountdown <= 0) { fireStepEvent(); state.stepCountdown = STEP_INTERVAL_MS; }
      // BACKWARD event
      state.backwardCountdown -= dt;
      if (state.backwardCountdown <= 0) { fireBackwardEvent(); state.backwardCountdown = BACKWARD_INTERVAL_MS; }

      if (!state.current) {
        state.nextFallIn -= dt;
        if (state.nextFallIn <= 0) spawnBlock();
      } else {
        state.nextFallIn -= dt;
        if (state.nextFallIn <= 0) { tryMove(0, 1); state.nextFallIn = state.fallInterval; }
      }
    }

    state.stepFlashTime = Math.max(0, state.stepFlashTime - dt);
    state.backwardFlashTime = Math.max(0, state.backwardFlashTime - dt);
    state.shakeT = Math.max(0, state.shakeT - dt);
    if (state.shakeT === 0) state.shakeAmt = 0;
    for (var pp of state.particles) { pp.x += pp.vx; pp.y += pp.vy; pp.vy += 0.15; pp.age += dt; }
    state.particles = state.particles.filter(function(x){ return x.age < x.maxAge; });
    for (var ff of state.floats) { ff.age += dt; ff.y -= dt * 0.03; }
    state.floats = state.floats.filter(function(x){ return x.age < x.maxAge; });

    if (opts.onScoreChange && !state.over) opts.onScoreChange({ score: state.score, alltimeBest: alltimeBest, timeLeft: state.timeLeft });

    draw();
    requestAnimationFrame(frame);
  }

  function draw() {
    var sx = 0, sy = 0;
    if (state.shakeAmt > 0) { sx = (rand()-0.5)*state.shakeAmt; sy = (rand()-0.5)*state.shakeAmt; }
    ctx.save();
    ctx.translate(sx, sy);
    ctx.clearRect(-20, -20, W + 40, H + 40);

    // Header
    ctx.fillStyle = "#333";
    ctx.font = "bold 15px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("OOM", W/2, 24);
    ctx.font = "10.5px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#888";
    ctx.fillText("← → move · space drop · gradients freed on step() · activations on backward", W/2, 42);

    // Phase indicator strip
    var phaseStr = state.stepFlashTime > 0
      ? "STEP() — gradients freeing"
      : (state.backwardFlashTime > 0 ? "BACKWARD — activations consuming" : "FORWARD — accumulating");
    var phaseColor = state.stepFlashTime > 0 ? "#3d9e5a"
                   : state.backwardFlashTime > 0 ? "#4a90c4" : "#888";
    ctx.fillStyle = phaseColor;
    ctx.font = "bold 11px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText(phaseStr, W/2, 60);

    // HBM border
    ctx.strokeStyle = "#a31f34";
    ctx.lineWidth = 2;
    ctx.strokeRect(hbmX, hbmY, hbmW, hbmH);

    // Grid lines (subtle)
    ctx.strokeStyle = "#f3f3f3";
    ctx.lineWidth = 0.5;
    for (var r = 1; r < rows; r++) { ctx.beginPath(); ctx.moveTo(hbmX, hbmY + r * cellSize); ctx.lineTo(hbmX + hbmW, hbmY + r * cellSize); ctx.stroke(); }
    for (var c = 1; c < cols; c++) { ctx.beginPath(); ctx.moveTo(hbmX + c * cellSize, hbmY); ctx.lineTo(hbmX + c * cellSize, hbmY + hbmH); ctx.stroke(); }

    // Placed blocks
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        if (!grid[r][c]) continue;
        ctx.fillStyle = grid[r][c].color;
        ctx.strokeStyle = grid[r][c].stroke;
        ctx.lineWidth = 1.5;
        ctx.fillRect(hbmX + c*cellSize + 1, hbmY + r*cellSize + 1, cellSize - 2, cellSize - 2);
        ctx.strokeRect(hbmX + c*cellSize + 1, hbmY + r*cellSize + 1, cellSize - 2, cellSize - 2);
      }
    }

    // Current falling block
    if (state.current) {
      var b = state.current;
      for (var i = 0; i < b.type.cells.length; i++) {
        var dc = b.type.cells[i][0], dr = b.type.cells[i][1];
        ctx.fillStyle = b.type.color;
        ctx.strokeStyle = b.type.stroke;
        ctx.lineWidth = 1.5;
        ctx.fillRect(hbmX + (b.col + dc)*cellSize + 1, hbmY + (b.row + dr)*cellSize + 1, cellSize - 2, cellSize - 2);
        ctx.strokeRect(hbmX + (b.col + dc)*cellSize + 1, hbmY + (b.row + dr)*cellSize + 1, cellSize - 2, cellSize - 2);
      }
    }

    // Step countdown bar (right side)
    var bx = hbmX + hbmW + 20, by = hbmY, bw = 12, bh = hbmH;
    ctx.fillStyle = "#eee";
    ctx.fillRect(bx, by, bw, bh);
    var stepFrac = Math.max(0, state.stepCountdown / STEP_INTERVAL_MS);
    ctx.fillStyle = "#3d9e5a";
    ctx.fillRect(bx, by + bh * (1 - stepFrac), bw, bh * stepFrac);
    ctx.fillStyle = "#555";
    ctx.font = "9px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("step", bx, by - 4);

    // Particles
    for (var pi = 0; pi < state.particles.length; pi++) {
      var pa = state.particles[pi];
      ctx.globalAlpha = Math.max(0, 1 - pa.age / pa.maxAge);
      ctx.fillStyle = pa.color;
      ctx.beginPath(); ctx.arc(pa.x, pa.y, 2, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Floats
    for (var fi = 0; fi < state.floats.length; fi++) {
      var ff = state.floats[fi];
      ctx.globalAlpha = Math.max(0, 1 - ff.age / ff.maxAge);
      ctx.fillStyle = ff.color;
      ctx.font = "bold 12px 'Helvetica Neue', Arial, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(ff.text, ff.x, ff.y);
    }
    ctx.globalAlpha = 1;

    // Legend (left side)
    var lx = 20, ly = hbmY;
    ctx.textAlign = "left";
    var legendData = [
      { c: "#cfe2f3", s: "#4a90c4", n: "activation" },
      { c: "#f9d6d5", s: "#c44",    n: "gradient" },
      { c: "#fdebd0", s: "#c87b2a", n: "optimizer" },
      { c: "#d4edda", s: "#3d9e5a", n: "KV cache" }
    ];
    for (var i = 0; i < legendData.length; i++) {
      ctx.fillStyle = legendData[i].c;
      ctx.strokeStyle = legendData[i].s;
      ctx.fillRect(lx, ly + i * 20, 10, 10);
      ctx.strokeRect(lx, ly + i * 20, 10, 10);
      ctx.fillStyle = "#555";
      ctx.font = "9px 'Helvetica Neue', Arial, sans-serif";
      ctx.fillText(legendData[i].n, lx + 14, ly + i * 20 + 8);
    }

    // HUD
    ctx.font = "11px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#333";
    ctx.textAlign = "left";
    ctx.fillText("score " + state.score, 20, H - 26);
    var secs = Math.ceil(state.timeLeft / 1000);
    ctx.textAlign = "center";
    ctx.fillStyle = secs <= 10 ? "#c44" : "#333";
    ctx.font = "bold 13px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("⏱ " + secs + "s", W/2, H - 26);
    ctx.fillStyle = "#333";
    ctx.font = "11px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("alltime best " + alltimeBest, W - 20, H - 26);
    ctx.font = "9px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#999";
    ctx.textAlign = "left";
    ctx.fillText("daily " + today + "  · space = hard drop · R = retry", 20, H - 10);

    if (state.over) drawGameOver();
    ctx.restore();
  }

  function drawGameOver() {
    ctx.fillStyle = "rgba(255,255,255,0.93)";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = state.timeLeft > 0 ? "#a31f34" : "#3d9e5a";
    ctx.font = "bold 28px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(state.timeLeft > 0 ? "OOM" : "🏆 survived!", W/2, H/2 - 24);
    ctx.fillStyle = "#333";
    ctx.font = "14px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("placed " + state.score + " tensors · best " + alltimeBest, W/2, H/2 + 4);
    ctx.fillStyle = "#777";
    ctx.font = "italic 11px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("tap or press R to retry", W/2, H/2 + 28);
  }

  requestAnimationFrame(frame);

  return {
    id: "oom",
    name: "OOM",
    ahaLabel: "You just played at",
    ahaText: "GPU memory management with lifetimes. Activations live from forward pass to backward; gradients clear when the optimizer step fires; optimizer state and KV cache persist. Real systems fight fragmentation with caching allocators, and trade compute for memory via activation checkpointing and offload. The shape of memory pressure — different lifetimes, different reclamation events — is what real frameworks juggle every step.",
    buildShareText: function(r) {
      return "MLSys Playground · OOM · " + today + "\n" +
             "packed " + r.score + " tensors before crash\n" +
             "play → mlsysbook.ai/games/oom/";
    }
  };
};
