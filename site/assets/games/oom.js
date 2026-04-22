/* ============================================================
   MLSys Playground — OOM
   ------------------------------------------------------------
   Tensor Tetris. Coloured blocks fall into the GPU's HBM
   region. Move left/right with arrow keys before they land.
   Fill the region without overflowing the top.

   Block types:
     - Activation (blue, wide): typical for forward pass
     - Gradient (red, narrow): spawns later (backward-ish vibe)
     - Optimizer state (orange, square): Adam-esque bulk
     - KV cache (green, tall): looming memory hog

   Score = blocks placed. Game over when a block can't fit.
   ============================================================ */

window.MLSP = window.MLSP || {};
MLSP.games = MLSP.games || {};

MLSP.games.oom = function(canvas, opts) {
  opts = opts || {};
  var ctx = canvas.getContext("2d");
  var W = canvas.width, H = canvas.height;

  // HBM region
  var hbmX = W/2 - 160, hbmY = 70, hbmW = 320, hbmH = H - 170;
  var cellSize = 20;
  var cols = Math.floor(hbmW / cellSize);   // 16 cols
  var rows = Math.floor(hbmH / cellSize);

  // Block shapes (in cells). Small shapes, think Tetris but simpler.
  var BLOCK_TYPES = [
    { name: "activation",       color: "#cfe2f3", stroke: "#4a90c4", cells: [[0,0],[1,0],[2,0]] },          // 3x1
    { name: "activation",       color: "#cfe2f3", stroke: "#4a90c4", cells: [[0,0],[1,0]] },                // 2x1
    { name: "gradient",         color: "#f9d6d5", stroke: "#c44",    cells: [[0,0],[0,1]] },                // 1x2
    { name: "optimizer state",  color: "#fdebd0", stroke: "#c87b2a", cells: [[0,0],[1,0],[0,1],[1,1]] },    // 2x2
    { name: "KV cache",         color: "#d4edda", stroke: "#3d9e5a", cells: [[0,0],[0,1],[0,2]] }           // 1x3
  ];

  // Grid: 0 = empty, color string = occupied
  var grid = [];
  for (var r = 0; r < rows; r++) {
    var row = [];
    for (var c = 0; c < cols; c++) row.push(null);
    grid.push(row);
  }

  var rand = Math.random;
  var state = {
    current: null,     // { type, row, col }
    nextFallIn: 500,
    fallInterval: 700,
    score: 0,
    over: false,
    shakeAmt: 0, shakeT: 0,
    floats: [], particles: []
  };

  var alltimeBest = MLSP.bestScore.get("oom");

  function spawnBlock() {
    var type = BLOCK_TYPES[Math.floor(rand() * BLOCK_TYPES.length)];
    var block = { type: type, row: 0, col: Math.floor(cols / 2) - 1 };
    // Fit check at spawn — if it doesn't fit, game over
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
      grid[b.row + dr][b.col + dc] = b.type;
    }
    state.score++;
    addFloat(W/2, hbmY - 10, "+1 " + b.type.name, b.type.stroke);
    // Speed up slightly
    state.fallInterval = Math.max(200, 700 - state.score * 10);
    // Clear any full rows (Tetris line-clear)
    for (var r = rows - 1; r >= 0; r--) {
      var full = true;
      for (var c = 0; c < cols; c++) if (!grid[r][c]) { full = false; break; }
      if (full) {
        grid.splice(r, 1);
        var empty = [];
        for (var c = 0; c < cols; c++) empty.push(null);
        grid.unshift(empty);
        state.score += 3;
        addFloat(hbmX + hbmW/2, hbmY + r * cellSize, "+3 freed!", "#3d9e5a");
        burst(hbmX + hbmW/2, hbmY + r * cellSize, "#3d9e5a", 20);
        r++;
      }
    }
    state.current = null;
    state.nextFallIn = 400;
  }

  function tryMove(dc, dr) {
    if (!state.current || state.over) return;
    var b = state.current;
    if (!collides(b, b.row + dr, b.col + dc)) {
      b.row += dr;
      b.col += dc;
    } else if (dr > 0) {
      // hit floor/block — lock
      lockBlock();
    }
  }

  function hardDrop() {
    if (!state.current || state.over) return;
    var b = state.current;
    while (!collides(b, b.row + 1, b.col)) b.row++;
    lockBlock();
  }

  function endGame() {
    if (state.score > alltimeBest) {
      alltimeBest = state.score;
      MLSP.bestScore.set("oom", alltimeBest);
    }
    if (opts.onGameOver) opts.onGameOver({
      score: state.score,
      alltimeBest: alltimeBest
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
    // Tap on left half = move left, right half = move right, below current = drop
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
      if (!state.current) {
        state.nextFallIn -= dt;
        if (state.nextFallIn <= 0) spawnBlock();
      } else {
        state.nextFallIn -= dt;
        if (state.nextFallIn <= 0) {
          tryMove(0, 1);
          state.nextFallIn = state.fallInterval;
        }
      }
    }

    state.shakeT = Math.max(0, state.shakeT - dt);
    if (state.shakeT === 0) state.shakeAmt = 0;
    for (var pp of state.particles) { pp.x += pp.vx; pp.y += pp.vy; pp.vy += 0.15; pp.age += dt; }
    state.particles = state.particles.filter(function(x){ return x.age < x.maxAge; });
    for (var ff of state.floats) { ff.age += dt; ff.y -= dt * 0.03; }
    state.floats = state.floats.filter(function(x){ return x.age < x.maxAge; });

    if (opts.onScoreChange && !state.over) opts.onScoreChange({ score: state.score, alltimeBest: alltimeBest });

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
    ctx.fillText("pack tensors into HBM · ← → to move · space to drop · full row = free memory", W/2, 42);

    // HBM label
    ctx.font = "bold 10px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#a31f34";
    ctx.textAlign = "left";
    ctx.fillText("HBM · " + (cols * rows) + " cells", hbmX, hbmY - 8);

    // HBM border
    ctx.strokeStyle = "#a31f34";
    ctx.lineWidth = 2;
    ctx.strokeRect(hbmX, hbmY, hbmW, hbmH);

    // Grid cells (very faint)
    ctx.strokeStyle = "#f0f0f0";
    ctx.lineWidth = 0.5;
    for (var r = 1; r < rows; r++) {
      ctx.beginPath(); ctx.moveTo(hbmX, hbmY + r * cellSize); ctx.lineTo(hbmX + hbmW, hbmY + r * cellSize); ctx.stroke();
    }
    for (var c = 1; c < cols; c++) {
      ctx.beginPath(); ctx.moveTo(hbmX + c * cellSize, hbmY); ctx.lineTo(hbmX + c * cellSize, hbmY + hbmH); ctx.stroke();
    }

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

    // Particles
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

    // HUD
    ctx.font = "11px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#333";
    ctx.textAlign = "left";
    ctx.fillText("score " + state.score, 20, H - 26);
    ctx.textAlign = "right";
    ctx.fillText("alltime best " + alltimeBest, W - 20, H - 26);
    ctx.font = "9px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#999";
    ctx.textAlign = "left";
    ctx.fillText("space = hard drop · R = retry", 20, H - 10);

    // Legend
    var lx = 20, ly = hbmY;
    ctx.textAlign = "left";
    for (var i = 0; i < BLOCK_TYPES.length; i++) {
      var b = BLOCK_TYPES[i];
      ctx.fillStyle = b.color;
      ctx.strokeStyle = b.stroke;
      ctx.fillRect(lx, ly + i * 20, 10, 10);
      ctx.strokeRect(lx, ly + i * 20, 10, 10);
      ctx.fillStyle = "#555";
      ctx.font = "9px 'Helvetica Neue', Arial, sans-serif";
      ctx.fillText(b.name, lx + 14, ly + i * 20 + 8);
    }

    if (state.over) drawGameOver();
    ctx.restore();
  }

  function drawGameOver() {
    ctx.fillStyle = "rgba(255,255,255,0.92)";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#a31f34";
    ctx.font = "bold 28px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("OOM", W/2, H/2 - 24);
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
    ahaText: "GPU memory management. Every training step keeps activations (for backward pass), gradients, optimizer states, and KV cache resident in HBM simultaneously. Run out and you crash. Real frameworks do this spatial packing for you with fragmentation, checkpointing, and offloading — and the consequences are the same: fill the wrong way, you OOM.",
    buildShareText: function(r) {
      return "MLSys Playground · OOM\n" +
             "packed " + r.score + " tensors before crash\n" +
             "play → mlsysbook.ai/games/oom/";
    }
  };
};
